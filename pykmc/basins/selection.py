import numpy as np
from .exit_time_solver import BisectionSolver
from .connectivity import StatesConnectivity
from .utils import solve_master_equation
from pykmc.result import (
    Result,
    Ok,
    ErrorInfo,
    BasinSelectorOutput,
    BasinExitTimeSolverOutput,
)

# TODO : Use a Abstract Selector (if implement a new one, eg MRT)
# TODO : For the moment spectral decomposition=True is hardcoded, and it is assumed that we use BisectionSolver, need to modify if use different (and use builder)


class FPTASelector:
    """
    Selector implementing First Passage Time Analysis (FPTA) to determine the exit time and exit transition of a basin.

    This class follows the procedure described in Ref. [1, 2]:

        1. Build the generator matrix over the transient states, with every absorbing state collapsed into one.
        2. Use a numerical solver to compute the exit time from it.
        3. Given the exit time, compute the absorbed population each transition leaving the basin carries, and select the exit transition.

    Attributes
    ----------
    M : np.ndarray or None
        Generator over the transient states plus one absorbing row, which
        collects every transition leaving the basin.
    order : dict[int, int] or None
        Transient state to matrix row.

    References
    ----------
    [1] doi.org/10.1063/1.3369627
    [2] doi.org/10.1063/5.0015039
    """

    def __init__(self) -> None:

        self.M = None  # Reduced absorbing Markov chain generator matrix
        self.order = None  # Transient state to matrix row

    def select_from_connectivity(
        self, connectivity_table: StatesConnectivity, entry: int = 0
    ) -> Result[BasinSelectorOutput, ErrorInfo]:
        """
        Find both an exit time and the transition the basin is left by, from a `StatesConnectivity` object.

        Parameters
        ----------
        connectivity_table : StatesConnectivity
            StatesConnectivity object.
        entry : int
            The state the basin was entered from; the chain starts there.

        Returns
        -------
        Result[BasinSelectorOutput, ErrorInfo]
            - Ok(BasinSelectorOutput(t_exit, exit_row)) on success.
            - Err(ErrorInfo) if exit time solver failed.
        """
        self.build_matrix(connectivity_table)

        # Find exit time :
        result = self.get_exit_time(self.order[entry])
        if not result.is_ok():  # Solver Err when determining t_exit
            return result
        t_exit = result.ok_value().t_exit

        # Find exit transition
        exit_row = self.select_exit_row(connectivity_table, t_exit, self.order[entry])

        return Ok(BasinSelectorOutput(t_exit=t_exit, exit_row=exit_row))

    def exit_shares(self, connectivity_table: StatesConnectivity, entry: int = 0) -> np.ndarray:
        """Each exit's probability of being the one the basin is left by, in `connectivity_table.exits()` order.

        The share of an exit is its rate times the expected time the chain
        spends in the state it leaves before absorbing, k_e * tau_u.
        """
        self.build_matrix(connectivity_table)
        n_transient_states = len(self.order)

        p0 = np.zeros(n_transient_states)
        p0[self.order[entry]] = 1
        tau = np.linalg.solve(self.M[:n_transient_states, :n_transient_states], p0)

        exits = connectivity_table.exits()
        shares = exits["k_forward"].to_numpy(dtype=float) * tau[exits["state"].map(self.order).to_numpy()]
        return shares

    def build_matrix(self, connectivity_table: StatesConnectivity) -> None:
        """
        Construct the generator matrix M, every absorbing state collapsed into its last row.

        The matrix is defined as:
            - M_ij = -k_ji for i ≠ j
            - M_ii = -sum_{j≠i} M_ij
        where k are the rates.

        A state is transient when something was explored out of it, so it
        appears in the `state` column; the absorbing row has no outgoing
        rate, so its column is 0.

        Parameters
        ----------
        connectivity_table : StatesConnectivity
            StatesConnectivity object with forward/backward rates.

        Returns
        -------

        None
        """
        df = connectivity_table.df
        self.order = {state: row for row, state in enumerate(sorted(set(df["state"])))}
        n_transient_states = len(self.order)
        self.M = np.zeros((n_transient_states + 1, n_transient_states + 1))

        # Non diagonal elements : M_ij = -k_ji
        columns = df["state"].map(self.order).to_numpy()
        rows = df["state_connexion"].map(self.order).fillna(n_transient_states).to_numpy(dtype=int)
        np.subtract.at(self.M, (rows, columns), df["k_forward"].to_numpy(dtype=float))

        # Diagonal elements : M_ii = sum_j k_ij
        for i in range(n_transient_states):
            self.M[i, i] = 0.0
            self.M[i, i] = -self.M[:, i].sum()

    def get_exit_time(self, entry_row: int = 0) -> Result[BasinExitTimeSolverOutput, ErrorInfo]:
        """
        Use Solver to find the exit time form the reduced matrix.

        Parameters
        ----------
        entry_row : int
            Matrix row of the state the basin was entered from.

        Returns
        -------
        Result[BasinExitTimeSolverOutput, ErrorInfo]
            - Ok(result) containing t_exit on success.
            - Err(ErrorInfo) if solver failed.
        """

        # Initialize
        p0 = np.zeros(len(self.M))
        p0[entry_row] = 1  # the chain starts where the basin was entered

        # Pick random number between [0,1) representing the probability of being in an absorbing states after time t
        r1 = np.random.random()

        # Use solver :
        exit_time_solver = BisectionSolver(self.M, p0, r1)
        result = exit_time_solver.solve()

        return result

    def select_exit_row(
        self, connectivity_table: StatesConnectivity, t_exit: float, entry_row: int = 0
    ) -> int:
        """
        Find which transition the basin is left by at the given exit time.

        A transition's share of the absorbed population is its rate times how
        long the chain sits in the state it leaves, k_e * tau_u. Summed over the
        transitions reaching one absorbing state that is exactly that state's
        population at `t_exit`, so drawing a transition also draws the state it
        reaches, with the distribution the absorbing populations give.

        Parameters
        ----------
        connectivity_table : StatesConnectivity
            StatesConnectivity object.
        t_exit : float
            Exit time.
        entry_row : int
            Matrix row of the state the basin was entered from.

        Returns
        -------
        int
            The connectivity row of the transition selected.
        """
        n_transient_states = len(self.order)
        p0 = np.zeros(len(self.M))
        p0[entry_row] = 1  # the chain starts where the basin was entered

        # p(t) = exp(-M t) p0
        p = np.real(solve_master_equation(self.M, t_exit, p0))

        # dp/dt = -M p integrates to M tau = p(0) - p(t) over the transient
        # block, so one solve gives every transient state's time-integrated
        # occupancy. A residence time cannot be negative; the solve can still
        # return a small negative where a population has all but drained.
        transient = slice(None, n_transient_states)
        tau = np.linalg.solve(self.M[transient, transient], p0[transient] - p[transient])
        tau = np.clip(tau, 0.0, None)

        exits = connectivity_table.exits()
        flux = np.cumsum(exits["k_forward"].to_numpy(dtype=float) * tau[exits["state"].map(self.order).to_numpy()])

        # Drawing against the running flux rather than a normalized one keeps
        # the draw strictly inside the last interval.
        r2 = np.random.random()
        exit_row = int(exits.index[np.searchsorted(flux, r2 * flux[-1])])
        return exit_row
