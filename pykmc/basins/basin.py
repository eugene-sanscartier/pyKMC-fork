from .detection import Detector
from .exploration import Explorer, BasinGenericEventExplorer
from .connectivity import BasinStatesConnectivity
from .selection import FPTASelector
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pykmc import (
    System,
    Parameters,
    Configuration,
    NeighborsList,
    AtomicEnvironment,
    ReferenceEventTable,
    PointSetRegistration,
    check_match,
    Reconstruction,
)
from typing import Optional
from ..utils import geometry
from ..log import fmt_hash, fmt_distance, fmt_energy, fmt_error, fmt_number, fmt_rate, fmt_time
from .. import log
from ..rate_constant import compute_rate_Eyring
import pandas as pd
import copy
import numpy as np
from scipy.spatial import cKDTree
from pykmc.result import Ok, Err, ErrorInfo, ErrorType, BasinOutput, Result, ShapeID

# TODO: StateDate is here to handle state informations, when State Object will be creates, need to remove
# TODO: For the moment Basin uses EnergyThresholdDetector, BasinGenericEventExplorer, FPTASelector, need to deal with possible multiple implementation with builder.
# TODO: Could think of refining transient -> absorbing event when exploring
# TODO : Exit if state 0 leads to all absorbing states because all unknown environments, here FTPA fails but because only have 1 transient state (0), should be a different ERROR.TYPE
# TODO should also check if we apply same event to different central atoms but same saddle position meaning that it s a duplicate event, so remove.

# Two basin states are the same one when every atom of each is this close (A)
# to some atom of the other.
STATE_MATCH_TOL = 0.3


@dataclass
class PlacedEvent:
    """A generic event pasted onto a state, awaiting its relaxation.

    `system` already sits at the pasted positions -- the final ones under the
    `global` style, the saddle ones under `global/reconstruction`.
    """

    system: System
    initial_configuration: Configuration
    final_configuration: Configuration
    neighbors: np.ndarray


@dataclass
class StateData:
    system: Optional[System]
    environment: Optional[AtomicEnvironment]
    neighbors_list: Optional[NeighborsList]
    transient: bool = False
    visited: bool = False
    # Every non-crystal atom's resolved ShapeID, set by unknown_environment()
    # only once the whole state passes the gate; None means "not
    # established", never "no shapes". Survives release_heavy_objects():
    # nothing during basin exploration can invalidate it, since no search
    # runs and so the shape catalog cannot change.
    atom_shapes: Optional[dict[int, ShapeID]] = None
    _tree: Optional[cKDTree] = field(default=None, repr=False, compare=False)

    @property
    def tree(self) -> cKDTree:
        """Periodic KD-tree over this state's own positions, built once."""
        if self._tree is None:
            self._tree = cKDTree(
                self.system.positions, boxsize=np.diag(self.system.cell).tolist()
            )
        return self._tree

    def release_heavy_objects(self) -> None:
        """Release heavy objects"""
        self.neighbors_list = None
        self.environment = None

    def ensure_full_state(self, params: Parameters) -> None:
        if self.system is not None:
            if self.neighbors_list is None:
                self.neighbors_list = NeighborsList(
                    self.system,
                    params.atomicenvironment.rnei,
                    params.atomicenvironment.rcut,
                    params.atomicenvironment.rnei_pairs,
                )
            if self.environment is None:
                self.environment = AtomicEnvironment(
                    params.atomicenvironment.style,
                    self.neighbors_list.neighbors_list["rnei"],
                    self.neighbors_list.neighbors_list["rcut"],
                    params.atomicenvironment.neighbors_add,
                    coordination_threshold=params.atomicenvironment.coordination_threshold,
                    types=self.system.types,
                    coloring_mode=params.atomicenvironment.coloring_mode,
                )


class BasinsGenericEvents:
    def __init__(self, params: Parameters, reference_table, manager) -> None:
        self.params = params  # Parameters object with basins parameters
        self.explorer = None  # object to explore a state in the basin
        self.reference_table = reference_table  # Object with reference generic events
        self.manager = manager  # object to do external task (minimize, refine)

        self.connectivity_table = None  # Dataframe of basin connexion state
        self.selected_event = None  # The selected event after basin exploration
        self.entry = None  # State the basin was entered from, tracked across reindexing
        self.current_state = None  # Current state where we're at
        self.states_to_explore = None  # List of state to explore
        self.explored_states = None  # List of state that we already explored
        self.next_state_index = 1  # First state index never handed out yet
        self.states: dict[int, StateData] = {}  # Dictionnary of StateDate
        # Keyed by connectivity row, not by (state, state_connexion): several
        # transitions can join the same pair of states, on different atoms.
        self.absorbing_saddle_configurations: dict[int, Configuration] = {}

    def execute(self, system):
        """
        run the basin exploration and select an event from a system, corresponding to the first state in the basin, it is assumed that this state is transient.
        """
        # initialize the basin
        self._initialize(system)
        log.debug(f"Basin: entered at state {self.current_state}, style '{self.params.basin.style}'", depth=2)

        # A basin is many small independent engine calls -- one relaxation per
        # candidate state, one saddle refinement per exit -- so the whole of it
        # runs on the session pool, which does them several at a time, rather
        # than on the single global engine.
        self.manager.use_local()

        # Every state discovered *during* exploration gets this check inside
        # construct_connexion_table()'s loop before being explored -- but
        # state 0 is pre-added by _initialize() and never passes through
        # that branch, so it has to be checked here instead. Without this,
        # a state 0 with an uncatalogued atom silently explores to zero
        # connectivity entries, and reorder_states_index() then returns an
        # empty mapping that doesn't cover state 0's own index.
        self.states[0].ensure_full_state(self.params)
        unknown = self.unknown_environment(self.states[0])
        if unknown is not None:
            log.debug(f"Basin: entry state 0 rejected -- {unknown}", depth=2)
            return Err(
                ErrorInfo(
                    type=ErrorType.BASIN_UNKNOWN_INITIAL_ENVIRONMENT,
                    message="Basin: cannot be entered -- {}.".format(unknown),
                )
            )
        log.debug(
            f"Basin: entry state 0 known ({len(self.states[0].atom_shapes)} non-crystal atoms)",
            depth=2,
        )

        # explore the basin
        result = self.construct_connexion_table()
        if not result.is_ok():
            log.debug(f"Basin: exploration failed -- {fmt_error(result.err_value())}", depth=2)
            return result
        # reorder states index
        mapping = self.connectivity_table.reorder_states_index()
        self.states = {mapping[old]: val for old, val in self.states.items()}
        self.entry = mapping[self.entry]
        # Refine absorbing states
        result = self.refine_absorbing(system)
        if not result.is_ok():
            log.debug(f"Basin: exit refinement failed -- {fmt_error(result.err_value())}", depth=2)
            return result
        # apply selector algorithm to find t_exit and exit_state
        log.debug(f"Basin: solving mean exit time over {len(self.states)} states", depth=2)
        result = self.selector.select_from_connectivity(self.connectivity_table)
        if not result.is_ok():
            log.debug(f"Basin: exit selection failed -- {fmt_error(result.err_value())}", depth=2)
            return result
        # Construct output KMC needs
        t_exit = result.ok_value().t_exit
        exit_state = result.ok_value().exit_state

        # The exit is one row of the connectivity table, and the refined saddle
        # and barrier belong to that row: several rows can join the same pair of
        # states on different atoms, so the pair alone does not identify it.
        # Only the rows that leave the basin were refined, and rows into one
        # state can disagree about that, so the row is taken from those.
        exit_row = self.connectivity_table.get_transition_to_state(
            target_state=exit_state, as_tuples=False, only_exits=True
        )
        idx = exit_row.index[0]
        from_state, event_idx, central_atom, sym_idx, is_transient = (
            self.connectivity_table.to_tuples(exit_row)[0]
        )
        # Ensure from_state is state are full
        self.states[from_state].ensure_full_state(self.params)
        log.debug(
            f"Basin: exit {from_state}->{exit_state},"
            f" delr={fmt_distance(self.delr_from_entry(exit_state))} from entry {self.entry},"
            f" t_exit={fmt_time(t_exit)}",
            depth=2,
        )

        neighbors = self.states[from_state].neighbors_list.get_neighbors(
            "rcut", central_atom
        )
        saddle_configuration = self.absorbing_saddle_configurations[idx]
        return Ok(
            BasinOutput(
                initial_system_configuration=self.states[from_state].system.configuration,
                central_atom=central_atom,
                saddle_configuration=saddle_configuration,
                final_configuration=self.states[exit_state].system.configuration[neighbors],
                neighbors=neighbors,
                dE_forward=self.connectivity_table.df.loc[idx, "dE_forward"],
                k_tot=self.connectivity_table.df.loc[
                    self.connectivity_table.df["transient"] == False, "k_forward"
                ].sum(),
                t_exit=t_exit,
                exit_state=exit_state,
                from_state=from_state,
                num_reference_event=event_idx,
            )
        )

    def delr_from_entry(self, state: int) -> float:
        """Largest per-atom displacement (A) between `state` and the state the basin was entered from."""
        return geometry.compute_delr_max(
            self.states[self.entry].system.configuration,
            self.states[state].system.configuration,
        )

    def _initialize(self, system) -> None:
        """
        Initialize necessary component after entering in basin. We always enter in state == 0.
        """
        self.entry = 0
        self.current_state = 0
        self.states_to_explore = [0]
        self.explored_states = []
        # A state index is never reused: an index that has been merged away is
        # still in `explored_states`, so handing it out again gives a
        # transition a target nothing will ever create.
        self.next_state_index = 1
        self.connectivity_table = BasinStatesConnectivity()
        self.explorer = BasinGenericEventExplorer(
            params=self.params, reference_table=self.reference_table
        )
        self.selector = FPTASelector()
        new_system = System.from_configuration(
            system.configuration.copy(), pbc=system.pbc.copy()
        )
        self._add_state(
            state_index=0, system=new_system
        )  # add current state 0 to self.states

    def construct_connexion_table(self):
        """
        explore the basin and construct the connextion table
        """
        # Loop over state to explore
        while len(self.states_to_explore) != 0:
            # Every transition of the state just explored points at a state
            # that does not exist yet, and creating one is an engine
            # minimization -- so the whole set is created in one go, its
            # relaxations sharing the session pool, instead of one per
            # iteration.
            pending = [state for state in self.states_to_explore if state not in self.states]
            if len(pending) != 0:
                result = self.create_states(pending)
                if not result.is_ok():
                    return result
                continue

            # next state to explore :
            to_explore = self.states_to_explore[0]

            # Explore state
            self.current_state = to_explore
            start_index = self.next_state_index

            self.explorer.explore(
                state=self.states[to_explore],
                state_index=self.current_state,
                start_index=start_index,
            )
            n_transitions = len(self.explorer.connectivity_table.get_table())
            # The explorer numbers its transitions start_index, start_index+1,
            # ... so this is the first index it left free.
            self.next_state_index = start_index + n_transitions

            # to_explore has been explored :
            self.states_to_explore.remove(to_explore)
            self.explored_states.append(to_explore)

            # Merge state connectivity table to basin connectivity table
            self.connectivity_table.merge(self.explorer.connectivity_table)
            # Clrean explorer connectivity table
            self.explorer.clear()
            self.update_to_explore()
            log.status(
                f"state {to_explore:5d}", "EXPLORED",
                f"{n_transitions} transitions"
                f" (explored={len(self.explored_states)}, to_explore={len(self.states_to_explore)})",
            )
            # Clean heaby state object :
            self.states[to_explore].release_heavy_objects()

        self.log_spread()
        return Ok(None)

    def log_spread(self) -> None:
        """Write how far each explored state sits from the entry, as a displacement."""
        if not log.is_debug_enabled():
            return

        for state in sorted(self.states):
            log.status(
                f"state {state:5d}", "SPREAD",
                f"delr={fmt_distance(self.delr_from_entry(state))}",
            )

    def create_states(self, pending: list[int]):
        """Create every state of `pending`, relaxing them concurrently on the session pool.

        Placing a generic event is pure geometry and stays on this thread; the
        relaxation that follows is the engine call, and the whole wave's are in
        flight together. Results are then folded in `pending` order, so which
        candidate wins a duplicate pair does not depend on which minimization
        finished first.
        """
        origins = {}
        placements = {}

        for to_explore in log.progress(pending, len(pending), label="Basin placement", reflow=2):
            # find a state and an event from which we go to the state that we want to create
            from_state, event_idx, central_atom, sym_idx, is_transient = (
                self.connectivity_table.get_transition_to_state(target_state=to_explore)
            )
            result = self.place_generic_event(from_state, event_idx, central_atom, sym_idx)
            if not result.is_ok():
                log.status(
                    f"state {to_explore:5d}", "PLACE_FAIL",
                    f"from {from_state}, atom {central_atom:6d}, event {event_idx}, "
                    f"sym {sym_idx:3d}, {fmt_error(result.err_value())}",
                )
                return result
            origins[to_explore] = (from_state, is_transient)
            placements[to_explore] = result.ok_value()
            dE_fwd = self.reference_table.table.loc[
                self.reference_table.table["idx_ref"] == event_idx, "dE_forward"
            ].iloc[0]
            log.status(
                f"state {to_explore:5d}", "PLACED",
                f"from {from_state}, atom {central_atom:6d}, event {event_idx}, "
                f"sym {sym_idx:3d}, dE_fwd={fmt_energy(dE_fwd)}",
            )

        relaxed = self.relax_placements(placements)

        for to_explore in log.progress(pending, len(pending), label="Basin state creation", reflow=2):
            result = relaxed[to_explore]
            if not result.is_ok():
                log.status(f"state {to_explore:5d}", "RELAX_FAIL", fmt_error(result.err_value()))
                return result
            from_state, is_transient = origins[to_explore]
            self.add_relaxed_state(to_explore, result.ok_value(), is_transient)

        # A wave's states typically share one from_state, so its heavy objects
        # are dropped once the whole wave is placed.
        for from_state, _ in origins.values():
            self.states[from_state].release_heavy_objects()

        return Ok(None)

    def relax_placements(self, placements: dict[int, PlacedEvent]) -> dict[int, Result]:
        """Relax every placed event, all of them submitted before any result is read."""
        if self.params.basin.style == "global":
            log.debug(f"Basin: minimizing {len(placements)} placements", depth=2)
            futures = {
                to_explore: self.manager.minimize_with_results(
                    self.params, configuration=placed.system.configuration.copy()
                )
                for to_explore, placed in placements.items()
            }
            relaxed = {}
            for to_explore, placed in log.progress(
                placements.items(), len(placements), label="Basin minimization", reflow=2,
            ):
                min2_configuration, _ = futures[to_explore].result()
                placed.system.update_positions(min2_configuration)
                relaxed[to_explore] = Ok(placed.system)
                log.status(f"state {to_explore:5d}", "MINIMIZED")
            return relaxed

        elif self.params.basin.style == "global/reconstruction":
            log.debug(f"Basin: reconstructing {len(placements)} placements", depth=2)
            jobs = {}
            for to_explore, placed in placements.items():
                local_types = np.asarray(placed.system.types)[placed.neighbors]
                jobs[to_explore] = (
                    placed.initial_configuration.with_types(local_types),
                    placed.final_configuration.with_types(local_types),
                    placed.system.configuration,
                    placed.neighbors,
                )
            outcomes = Reconstruction(self.params, self.manager).reconstruct_many(
                jobs, self.params.psr.matching_score_thr
            )

            relaxed = {}
            for to_explore, placed in placements.items():
                result = outcomes[to_explore]
                if not result.is_ok():
                    relaxed[to_explore] = result
                    log.status(f"state {to_explore:5d}", "RECON_FAIL", fmt_error(result.err_value()))
                    continue
                placed.system.update_positions(result.ok_value().min2_configuration)
                relaxed[to_explore] = Ok(placed.system)
                log.status(
                    f"state {to_explore:5d}", "RECONSTRUCTED",
                    f"E={fmt_energy(result.ok_value().min2_etot)}",
                )
            return relaxed

        else:
            raise ValueError(f"Unknown {self.params.basin.style} style parameter.")

    def add_relaxed_state(self, state_index: int, new_system: System, is_transient: bool) -> None:
        """Register one relaxed state: merge it into the state it already is, or add it and decide whether it can be explored."""
        # Check if it is a new_system or already in states
        existing = self.is_new_state(new_system)
        if existing != -1:  # It already exists
            # update table
            self.connectivity_table.change_state_index(
                current_index=state_index, new_index=existing
            )
            log.status(f"state {state_index:5d}", "MERGED", f"into state {existing}")
            self.explored_states.append(state_index)
            self.states_to_explore.remove(state_index)
            return

        # add state
        self._add_state(state_index=state_index, system=new_system, transient=is_transient)
        if log.is_debug_enabled():
            log.status(
                f"state {state_index:5d}", "ADDED",
                f"total={len(self.states)}, to_explore={len(self.states_to_explore)}, "
                f"delr={fmt_distance(self.delr_from_entry(state_index))} from entry",
            )

        self.states[state_index].ensure_full_state(self.params)
        # Check if unknown atomic environments
        unknown = self.unknown_environment(self.states[state_index])
        if unknown is not None:
            # We consider that this state is an absorbing one because we need to search new events (in main KMC loop)
            # Need to update the connectivity table
            self.connectivity_table.change_state_to_absorbing(state_index)
            self.states[state_index].transient = False
            is_transient = False
            log.status(f"state {state_index:5d}", "ABSORBING", unknown)

        if not is_transient:
            self.states_to_explore.remove(state_index)
            self.explored_states.append(state_index)
            log.status(f"state {state_index:5d}", "EXIT", f"to_explore={len(self.states_to_explore)}")

        # `unknown_environment` has recorded every atom's shape on the state,
        # which is all the explorer reads -- so the neighbors list and
        # environment are released here rather than held until the state's
        # turn to be explored comes up, batches later.
        self.states[state_index].release_heavy_objects()

    def select_event(self):
        """
        select an event base on the selector algorithm
        """
        pass

    def get_seletec_event(self):
        """
        convinient method
        """
        pass

    def update_to_explore(self):
        # Find all state index in the connexion table :
        unique_states = set(self.connectivity_table.get_table()["state"]).union(
            set(self.connectivity_table.get_table()["state_connexion"])
        )
        self.states_to_explore = list(
            unique_states.difference(set(self.explored_states))
        )

    def place_generic_event(self, from_state, event_idx, central_atom, sym_idx):
        """Paste the generic event onto `from_state`, ready for `relax_placements` to relax.

        Pure geometry: PSR, the symmetry variant and the positions handed to
        the engine. No engine call happens here, so a whole batch of these can
        be prepared and then relaxed together.
        """

        ref_event = self.reference_table.table[
            self.reference_table.table["idx_ref"] == event_idx
        ]  # event where event_idx == idx_ref
        if ref_event.empty:
            raise ValueError(f"idx_ref={event_idx} not found in reference table")
        ref_event = ref_event.iloc[0].copy()
        #        ref_event = self.reference_table.table.iloc[event_idx].copy()

        initial_configuration = ref_event["initial_configuration"]
        final_configuration = ref_event["final_configuration"]
        saddle_configuration = ref_event["saddle_configuration"]

        # Apply the generic event to the current state

        # ENSURE FULL STATE FOR FROM STATE
        self.states[from_state].ensure_full_state(self.params)

        # We start from the from_state
        new_system = System.from_configuration(
            self.states[from_state].system.configuration.copy(), pbc=True
        )
        # new_system = copy.deepcopy(self.states[from_state].system)

        # Apply PSR between event initial position and environment positions of the central_atoms
        result = PointSetRegistration(
            self.params,
            new_system,
            ref_event,
            self.states[from_state].neighbors_list,
            central_atom,
        ).match()
        if not result.is_ok():  # PSR Err
            return result
            # Check if PointSetRegistration match is valid
        result = check_match(result, self.params.psr.matching_score_thr)
        if not result.is_ok():  # PSR matching score not valid :
            return result
        else:
            psr_output = result.ok_value()  # get psr results

        # Apply PSR to generic event to move

        # Apply symmetry matrix if sym != 0
        if sym_idx != 0:
            sym_matrix = ref_event["sym_matrix"][sym_idx]
            sym_perm = ref_event["sym_perm"][sym_idx]
            # sym_matrix is a rotation about the reference initial shape's own
            # centroid (ira_mod.SOFI's convention, see unique_symmetries), not
            # the coordinate origin -- pivot on that centroid rather than the
            # raw absolute positions, or the reconstructed shape comes out
            # wrong by roughly the cluster's offset from the origin.
            pivot = initial_configuration.positions.mean(axis=0)
            initial_configuration = geometry.transform_positions(
                initial_configuration - pivot, sym_matrix, 0, sym_perm, wrap=False,
            ) + pivot
            saddle_configuration = geometry.transform_positions(
                saddle_configuration - pivot, sym_matrix, 0, sym_perm, wrap=False,
            ) + pivot
            final_configuration = geometry.transform_positions(
                final_configuration - pivot, sym_matrix, 0, sym_perm, wrap=False,
            ) + pivot
        initial_configuration = geometry.transform_positions(
            initial_configuration,
            psr_output.rotation_matrix,
            psr_output.translation_matrix,
            psr_output.permutation_matrix,
        )
        saddle_configuration = geometry.transform_positions(
            saddle_configuration,
            psr_output.rotation_matrix,
            psr_output.translation_matrix,
            psr_output.permutation_matrix,
        )
        final_configuration = geometry.transform_positions(
            final_configuration,
            psr_output.rotation_matrix,
            psr_output.translation_matrix,
            psr_output.permutation_matrix,
        )

        # Move system do saddle positions
        neighbors = self.states[from_state].neighbors_list.get_neighbors(
            "rcut", central_atom
        )

        if self.params.basin.style == "global":
            new_system.update_positions(final_configuration, atom_idx=neighbors)

        elif self.params.basin.style == "global/reconstruction":
            new_system.update_positions(saddle_configuration, atom_idx=neighbors)

        else:
            raise ValueError(f"Unknown {self.params.basin.style} style parameter.")

        placed = PlacedEvent(
            system=new_system,
            initial_configuration=initial_configuration,
            final_configuration=final_configuration,
            neighbors=neighbors,
        )
        return Ok(placed)

    def refine_absorbing(self, system):
        """When connectivity table is build, and that we have dict of states, we refine the energy barrier and k_forward of the transient -> absorbing event"""
        # compute the energy of the state
        # for all row in connectivity table where we need to refine
        exits_df = self.connectivity_table.df[self.connectivity_table.df["transient"] == False]
        log.debug(f"Basin: refining {len(exits_df)} exits", depth=2)

        futures_context = {}  # idx → { "saddle": f_sad, ... }
        for idx, row in log.progress(
            exits_df.iterrows(), len(exits_df), label="Basin exit preparation", reflow=2,
        ):
            # tmp_system = copy.deepcopy(self.states[row["state"]].system)
            tmp_system = System.from_configuration(
                self.states[row["state"]].system.configuration.copy(), pbc=True
            )
            # move to generic saddle positions
            ref_event = self.reference_table.table[
                self.reference_table.table["idx_ref"] == row["event_connexion"]
            ]
            if ref_event.empty:
                raise ValueError(
                    f"idx_ref={row['event_connexion']} not found in reference table"
                )
            ref_event = ref_event.iloc[0].copy()
            # ref_event = self.reference_table.table.iloc[row["event_connexion"]].copy()
            saddle_configuration = ref_event["saddle_configuration"]
            # Apply PSR between event initial position and environment positions of the central_atoms

            # ENSURE "STATE" FULL
            self.states[row["state"]].ensure_full_state(self.params)

            result = PointSetRegistration(
                self.params,
                tmp_system,
                ref_event,
                self.states[row["state"]].neighbors_list,
                row["central_atom"],
            ).match()
            if not result.is_ok():  # PSR Err
                log.status(
                    f"exit {row['state']:5d}->{row['state_connexion']:<5d}", "ALIGN_FAIL",
                    f"atom {row['central_atom']:6d}, event {row['event_connexion']}, "
                    f"sym {row['sym']:3d}, {fmt_error(result.err_value())}",
                )
                return result
                # Check if PointSetRegistration match is valid
            matching_score_thr = self.params.psr.matching_score_thr + 0.25 * self.params.psr.matching_score_thr
            result = check_match(result, matching_score_thr)
            if not result.is_ok():  # PSR matching score not valid :
                log.status(
                    f"exit {row['state']:5d}->{row['state_connexion']:<5d}", "ALIGN_FAIL",
                    f"atom {row['central_atom']:6d}, event {row['event_connexion']}, "
                    f"sym {row['sym']:3d}, {fmt_error(result.err_value())}, "
                    f"matching_score_thr={fmt_number(matching_score_thr)}",
                )
                return result
            else:
                psr_output = result.ok_value()  # get psr results

            # Apply symmetry matrix if sym != 0
            if row["sym"] != 0:
                sym_matrix = ref_event["sym_matrix"][row["sym"]]
                sym_perm = ref_event["sym_perm"][row["sym"]]
                # sym_matrix is a rotation about the reference initial
                # shape's own centroid (ira_mod.SOFI's convention, see
                # unique_symmetries), not the coordinate origin or
                # saddle_configuration's own centroid -- pivot on the
                # initial shape's centroid rather than the raw absolute
                # positions, or the reconstructed saddle comes out wrong
                # by roughly the cluster's offset from the origin.
                pivot = ref_event["initial_configuration"].positions.mean(axis=0)
                saddle_configuration = geometry.transform_positions(
                    saddle_configuration - pivot, sym_matrix, 0, sym_perm, wrap=False,
                ) + pivot
            saddle_configuration = geometry.transform_positions(
                saddle_configuration,
                psr_output.rotation_matrix,
                psr_output.translation_matrix,
                psr_output.permutation_matrix,
            )
            neighbors = self.states[row["state"]].neighbors_list.get_neighbors(
                "rcut", row["central_atom"]
            )

            future2 = self.manager.partn_refine(
                self.params,
                row["central_atom"],
                configuration=tmp_system.configuration.copy(),
                saddle_idx=neighbors.copy(),
                saddle_positions=saddle_configuration.positions.copy(),
            )  # send copy not reference ! -- active_volume routing happens inside partn_refine

            # save future in context :
            futures_context[idx] = {
                "saddle": future2,
                "neighbors": neighbors,
                "state": row["state"],
                "target": row["state_connexion"],
                "central_atom": row["central_atom"],
            }
            log.status(
                f"exit {row['state']:5d}->{row['state_connexion']:<5d}", "SUBMITTED",
                f"for refinement | atom {row['central_atom']:6d}, event {row['event_connexion']}, "
                f"sym {row['sym']:3d}, expected dE_fwd={fmt_energy(row['dE_forward'])}",
            )

            # RELEASE MEMORY :
            self.states[row["state"]].release_heavy_objects()

        # modify connectivity table entry -- future2 holds the already-refined dE (E_saddle - E_min)
        for idx, ctx in log.progress(
            futures_context.items(), len(futures_context), label="Basin exit refinement", reflow=2,
        ):
            result_sad = ctx["saddle"].result()
            if not result_sad.is_ok():
                log.status(
                    f"exit {ctx['state']:5d}->{ctx['target']:<5d}", "REFINE_FAIL",
                    fmt_error(result_sad.err_value()),
                )
                return result_sad
            dE = result_sad.ok_value().E_saddle
            k = compute_rate_Eyring(dE, self.params)
            catalogue_dE = self.connectivity_table.df.loc[idx, "dE_forward"]
            log.status(
                f"exit {ctx['state']:5d}->{ctx['target']:<5d}", "REFINED",
                f"dE_fwd={fmt_energy(dE)} (catalogue {fmt_energy(catalogue_dE)}), k={fmt_rate(k)}",
            )

            # also save saddle configuration refined -- re-imaged around the
            # live central atom so every atom in the cluster is mutually
            # contiguous (a raw slice can leave an atom that drifted across
            # a periodic boundary during the saddle search off by a whole
            # cell vector relative to the rest of the cluster), matching how
            # every other local cluster in the codebase is normalized.
            central_atom_position = self.states[ctx["state"]].system.positions[
                ctx["central_atom"]
            ]
            self.absorbing_saddle_configurations[idx] = geometry.unwrap_around(
                result_sad.ok_value().saddle[ctx["neighbors"]], central_atom_position
            )
            # update connectivity table row
            self.connectivity_table.df.loc[idx, "dE_forward"] = dE
            self.connectivity_table.df.loc[idx, "k_forward"] = k
        return Ok(None)

    def is_new_state(self, system):
        """The index of the state `system` already is, or -1 if it is a new one.

        Every candidate is compared against every state, so this is
        quadratic in the basin's size and its per-comparison cost is what
        bounds how large a basin can be explored.
        """
        distances, _ = self.states[0].tree.query(
            system.configuration.positions, k=1, distance_upper_bound=STATE_MATCH_TOL
        )
        probe = np.flatnonzero(~np.isfinite(distances))

        for state_index, state_data in self.states.items():
            if self.are_structures_equivalent(system.configuration, state_data, probe):
                return state_index
        return -1

    def are_structures_equivalent(
        self,
        configuration: Configuration,
        state: StateData,
        probe: Optional[np.ndarray] = None,
        tol: float = STATE_MATCH_TOL,
    ):
        """Whether `configuration` is the same state as `state`: every atom within `tol` of one of its atoms."""
        other = state.system.configuration
        if len(configuration.positions) != len(other.positions):
            return False

        # In full coloring mode, two states with the same geometry but a different
        # species arrangement (e.g. an Fe/Ni swap) are distinct; in grey mode they merge.
        if self.params.atomicenvironment.coloring_mode == "full" and not np.array_equal(
            configuration.types, other.types
        ):
            return False

        # distance_upper_bound reports "no atom within tol" as inf, so all-finite
        # is the max-distance test without searching past the answer.
        if probe is not None and len(probe) > 0:
            distances, _ = state.tree.query(
                configuration.positions[probe], k=1, distance_upper_bound=tol
            )
            if not np.isfinite(distances).all():
                return False

        distances, _ = state.tree.query(
            configuration.positions, k=1, distance_upper_bound=tol
        )

        return bool(np.isfinite(distances).all())

    def unknown_environment(self, state: StateData) -> str | None:
        """Describe the first atom in `state` whose reactions aren't known yet, or return `None` -- and then record every atom's shape on `state`.

        An atom classified `"crystal"` needs no event and is treated as
        known without classification: under the `cna/graph`-family styles,
        a bulk/perfectly-coordinated atom never gets a real topology hash
        computed for it in the first place (see
        `AtomicEnvironment.compute_cnagraph`), so there is no shape for it
        to classify into.

        Every other atom must both resolve to a catalogued `ShapeID` (a
        coarse `id_initial` collision can leave an atom's real shape
        uncatalogued even though its `id_initial` has other, unrelated
        entries, so the live geometry is classified directly rather than
        checking `id_initial` presence alone) and have finished its own
        search round (`ShapeTable.needs_search`). Merely having been seen
        before is not enough: `KMC.resolve_new_shapes` mints a catalog entry
        for every non-crystal atom it looks at, so a shape nobody has
        searched yet still resolves. Basin exploration can only build
        connectivity from already-catalogued events -- no live ARTn search
        happens here -- so an atom the main KMC loop still intends to search
        makes this state absorbing, to be handed back to that loop. An atom
        whose finished search found nothing is known, not unknown: some
        shapes genuinely have no events, and they contribute no transitions
        rather than blocking exploration forever.

        Parameters
        ----------
        state : StateData
            The state to check. On a `None` return, its `atom_shapes` is set
            to every non-crystal atom's resolved `ShapeID`, ready for
            `BasinGenericEventExplorer.explore()` to use without
            reclassifying; otherwise it is left untouched, since the state is
            never explored.

        Returns
        -------
        str | None
            Which atom blocks exploration and why, or `None` if every atom
            is known.

        """
        shapes = self.reference_table.shapes
        atom_shapes: dict[int, ShapeID] = {}

        for atom_idx, id_initial in enumerate(
            state.environment.atomic_environment_list
        ):
            if id_initial == "crystal":
                continue
            shape = shapes.resolve_live_shape(
                state.system, state.neighbors_list, state.environment, atom_idx, mint=False
            )
            if shape is None:
                return "atom {} (id {}) matches no catalogued shape".format(
                    atom_idx, fmt_hash(id_initial)
                )
            if shapes.needs_search(shape):
                return "atom {} has shape {}#{}, whose own search is still '{}'".format(
                    atom_idx,
                    fmt_hash(shape.id),
                    shape.sid,
                    shapes.get_shape_knowledge(shape).status,
                )
            atom_shapes[atom_idx] = shape

        state.atom_shapes = atom_shapes
        return None

    def _add_state(
        self,
        state_index,
        system=None,
        transient=True,
        applicable_events=None,
        visited=False,
        full=False,
    ):
        """Add a new state in the `self.states` dictionnary."""
        # to fit typing
        neighbors_list = []
        atomic_environment = []

        if full == True:
            neighbors_list = NeighborsList(
                system,
                self.params.atomicenvironment.rnei,
                self.params.atomicenvironment.rcut,
                self.params.atomicenvironment.rnei_pairs,
            )
            atomic_environment = AtomicEnvironment(
                self.params.atomicenvironment.style,
                neighbors_list.neighbors_list["rnei"],
                neighbors_list.neighbors_list["rcut"],
                self.params.atomicenvironment.neighbors_add,
                coordination_threshold=self.params.atomicenvironment.coordination_threshold,
                types=system.types,
                coloring_mode=self.params.atomicenvironment.coloring_mode,
            )
        else:
            neighbors_list = None
            atomic_environment = None
        new_state = StateData(
            system=system,
            environment=atomic_environment,
            neighbors_list=neighbors_list,
            transient=transient,
            visited=visited,
        )

        self.states[state_index] = new_state
