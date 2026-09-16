"""Module to reconstruct an event from saddle positions"""

from pykmc.enginemanager.lmpi.pool import Manager
from pykmc import Parameters, Configuration
from pykmc.result import Result, Ok, Err, ReconstructionOutput, ErrorInfo, ErrorType
import numpy as np
from pykmc.utils.geometry import push_towards, compute_delr_max, wrap_configuration
from pykmc import log

# TODO: Use it in KMC
# TODO: Clean reconstruct/split the method


class Reconstruction:
    def __init__(self, params: Parameters, manager: Manager) -> None:
        self.params = params
        self.manager = manager  # Manager objet that can perform minimization and return minimized positions

    def reconstruct(
        self,
        supposed_min1: Configuration,
        supposed_min2: Configuration,
        saddle: Configuration,
        delr_thr,
        neighbors=None,
    ):
        """From a saddle point, try to reconstruct the event to see if it matches the
        supposed min1 and min2 positions, and that the to minima are connected.

        Since we generaly save only the atomic environment of the central atom
        we can specified neighbors which correspond to the list index of atoms in
        saddle positions that we need to modifie to go toward min pos.

        The reconstruction procede as follow :
        From the saddle positions
        Move the system toward the first minimum (with fraction)
        Minimize and compare minimized positions with supposed min1 positions
        same for min2


        Parameters
        ----------
        supposed_min1 : Configuration
            The hypothesized first minimum's types/positions/cell (typically a
            local, neighbor-indexed cluster sharing `saddle`'s cell).
        supposed_min2 : Configuration
            The hypothesized second minimum's types/positions/cell.
        saddle : Configuration
            The saddle point's types/positions/cell.
        delr_thr : _type_
            _description_
        neighbors : _type_, optional
            _description_, by default None
            typically the neighors list of the in the atomic environment of the atom on which we apply the event
        """
        jobs = {0: (supposed_min1, supposed_min2, saddle, neighbors)}
        return self.reconstruct_many(jobs, delr_thr)[0]

    def reconstruct_many(self, jobs, delr_thr) -> dict:
        """Reconstruct several events at once, one `Result` per key of `jobs`.

        Every min1 is submitted before any is read, then every min2 of the
        events whose min1 came back valid -- so on a session pool the
        minimizations of independent events overlap, while an event whose min1
        is already invalid still costs no min2.

        Parameters
        ----------
        jobs : dict
            key -> `(supposed_min1, supposed_min2, saddle, neighbors)`, each
            entry as :meth:`reconstruct` takes them.
        delr_thr : float
            Maximum per-atom deviation from a supposed minimum.
        """
        jobs = {
            key: (min1, min2, saddle, np.arange(len(saddle)) if neighbors is None else neighbors)
            for key, (min1, min2, saddle, neighbors) in jobs.items()
        }

        futures = {
            key: self.manager.minimize_with_results(self.params, configuration=self._pushed(saddle, min1, neighbors))
            for key, (min1, _, saddle, neighbors) in jobs.items()
        }

        results = {}
        min1_configurations = {}
        jobs_items = jobs.items()
        if len(jobs) > 1:
            jobs_items = log.progress(jobs_items, len(jobs), label="Reconstruction min1", reflow=2)
        for key, (min1, _, _, neighbors) in jobs_items:
            min1_configuration, _ = futures[key].result()

            delr1 = compute_delr_max(min1, wrap_configuration(min1_configuration)[neighbors])
            if delr1 > delr_thr:
                results[key] = Err(
                    ErrorInfo(
                        type=ErrorType.RECONSTRUCTION_INVALID_MIN1,
                        message="did not retreive initial minimum : delr1 = {}".format(delr1),
                        variables={"delr1": delr1},
                    )
                )
            else:
                min1_configurations[key] = min1_configuration

        futures = {
            key: self.manager.minimize_with_results(
                self.params, configuration=self._pushed(jobs[key][2], jobs[key][1], jobs[key][3])
            )
            for key in min1_configurations
        }

        min1_configuration_keys = min1_configurations
        if len(min1_configurations) > 1:
            min1_configuration_keys = log.progress(min1_configurations, len(min1_configurations), label="Reconstruction min2", reflow=2)
        for key in min1_configuration_keys:
            min2_configuration, min2_etot = futures[key].result()
            _, min2, saddle, neighbors = jobs[key]

            delr2 = compute_delr_max(min2, wrap_configuration(min2_configuration)[neighbors])
            if delr2 > delr_thr:
                results[key] = Err(
                    ErrorInfo(
                        type=ErrorType.RECONSTRUCTION_INVALID_MIN2,
                        message=f"did not retreive expected final minimum : delr2 = {delr2}",
                        variables={"delr2": delr2},
                    )
                )
            else:
                results[key] = Ok(
                    ReconstructionOutput(
                        min1_configuration=min1_configurations[key],
                        saddle_configuration=saddle,
                        min2_configuration=min2_configuration,
                        min2_etot=min2_etot,
                    )
                )

        return results

    def _pushed(self, saddle: Configuration, target: Configuration, neighbors) -> Configuration:
        """`saddle` with its `neighbors` nudged toward `target`, ready to minimize."""
        push_configuration = saddle.copy()
        push_configuration[neighbors] = push_towards(
            saddle[neighbors], target, fraction=self.params.reconstruction.push_fraction
        )
        return push_configuration
