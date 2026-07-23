"""Module implementing the Refinement class that deals with the event refinement procedure."""

from dataclasses import dataclass, field

from .result import (
    Result,
    EventRefinementOutput,
    ErrorInfo,
    ErrorType,
    Err,
    Ok,
    RefinementTask,
)
from .point_set_registration import PointSetRegistration, check_match
from .utils import geometry
from .config import Config
from .system import System
from .neighbors_list import NeighborsList
from .log import LogKMC
from .atomic_environment import AtomicEnvironment
from .enginemanager.lmpi.pool import Manager
import ase.geometry
import numpy as np
import pandas as pd
import concurrent.futures


@dataclass
class PreparedRefinementTask:
    """Prepared refinement task ready for submission or immediate completion."""

    task: RefinementTask
    min2_positions: np.ndarray | None
    reference_energy_barrier: float
    neighbors: np.ndarray
    immediate_result: Result[EventRefinementOutput, ErrorInfo] | None = None
    submit_kwargs: dict = field(default_factory=dict, repr=False)


class Refinement:
    """Perfrom event refinements and deal with results.

    Parameters
    ----------
    config : Config
        The configuration of the simulation.
    loggers : LogKMC
        The logger of the KMC simulation.
    system : System
        The atomic system.
    neighbors_list : NeighborsList
        The neighbors lists of the system.
    atomic_environment : AtomicEnvironment
        The atomic environment of the system.
    engine : Engine
        The engine to use for the refinement.

    """

    def __init__(
        self,
        config: Config,
        loggers: LogKMC,
        system: System,
        neighbors_list: NeighborsList,
        atomic_environment: AtomicEnvironment,
        manager: Manager,
    ) -> None:
        self.config = config
        self.loggers = loggers
        self.system = system
        self.neighbors_list = neighbors_list
        self.atomic_environment = atomic_environment
        self.manager = manager
        self.results = None
        self.tasks = []

    def execute(
        self,
        df_reference_events: pd.DataFrame,
        total_energy,
        existing_pairs: set[tuple[int, int]] | None = None,
    ) -> None:
        """Execute event refinements for each reference event in the df_reference_events dataframe.

        It stores the results of the event refinements in self.results.

        Parameters
        ----------
        df_reference_events : pd.DataFrame
            dataframe of reference events to refine.
        existing_pairs : set[tuple[int, int]] | None, optional
            `(atom_index, num_reference_event)` pairs already present in the
            active event table (carried over from the previous step). These
            are skipped during refinement.

        """
        tasks = self.build_tasks(
            df_reference_events, total_energy, existing_pairs=existing_pairs
        )
        self.loggers.info("log", "\t :=> Refining {} events".format(len(tasks)))
        self.tasks = tasks
        self.results = [None] * len(tasks)
        for task_id, result in self._run_tasks(tasks).items():
            self.results[task_id] = result

    def build_tasks(
        self,
        df_reference_events: pd.DataFrame,
        total_energy: float,
        existing_pairs: set | None = None,
    ) -> list[RefinementTask]:
        """Build refinement tasks with stable ids and rerun context."""
        existing_pairs = existing_pairs or set()
        raw_tasks = []
        task_id = 0
        supposed_ktot = 0.0
        for _idx, dfevent in df_reference_events.iterrows():
            atoms_refine_idx = self.atomic_environment.get_atoms_with_id(
                dfevent["id_initial"]
            )
            ref_idx = int(dfevent["idx_ref"])
            for at_idx in atoms_refine_idx:
                if (at_idx, ref_idx) in existing_pairs:
                    continue
                for symmetry_index, _sym in enumerate(dfevent.at["sym_matrix"]):
                    raw_tasks += [(task_id, at_idx, dfevent, symmetry_index)]
                    supposed_ktot += dfevent.at["k"]
                    task_id += 1
        e_thr = self._get_energy_threshold(df_reference_events, supposed_ktot)
        return [
            RefinementTask(
                task_id=task_id,
                central_atom_index=at_idx,
                num_reference_event=dfevent["idx_ref"],
                symmetry_index=symmetry_index,
                dfevent=dfevent,
                total_energy=total_energy,
                e_thr=e_thr,
            )
            for task_id, at_idx, dfevent, symmetry_index in raw_tasks
        ]

    def retry(self, retry_task_ids: list[int]) -> None:
        """Rerun only the requested refinement jobs."""
        if not retry_task_ids:
            return
        retry_tasks = [self.tasks[task_id] for task_id in retry_task_ids]
        for task_id, result in self._run_tasks(retry_tasks).items():
            self.results[task_id] = result

    def _run_tasks(
        self,
        tasks: list[RefinementTask],
    ) -> dict[int, Result[EventRefinementOutput, ErrorInfo]]:
        if not tasks:
            return {}

        future_to_prepared = {}
        for task in tasks:
            prepared = self._prepare_task(task)
            future = self._submit_task(prepared)
            future_to_prepared[future] = prepared

        run_results = {}
        for i, future in enumerate(concurrent.futures.as_completed(future_to_prepared)):
            prepared = future_to_prepared[future]
            try:
                res = future.result()
            except Exception as exc:
                self.loggers.error(
                    "log",
                    f"\n\t task {prepared.task.task_id:5d} | atom {prepared.task.central_atom_index:6d} | {'RAISE':<5} type={type(exc).__name__}",
                )
                raise
            self.loggers.progress_bar("progress", i + 1, len(tasks))
            run_results[prepared.task.task_id] = self._finalize_result(res, prepared)
        return run_results

    def _prepare_task(
        self,
        task: RefinementTask,
    ) -> PreparedRefinementTask:
        at_idx = task.central_atom_index
        num_reference_event = task.num_reference_event
        symmetry_index = task.symmetry_index
        dfevent = task.dfevent

        neighbors = self.neighbors_list.get_neighbors("rcut", at_idx).copy()
        result_psr = PointSetRegistration(
            self.config, self.system, dfevent, self.neighbors_list, at_idx
        ).match()
        result_psr = check_match(result_psr, self.config.psr.matching_score_thr)
        if not result_psr.is_ok():
            return PreparedRefinementTask(
                task=task,
                min2_positions=None,
                reference_energy_barrier=dfevent["dE_forward"],
                neighbors=neighbors,
                immediate_result=result_psr,
            )

        output_psr = result_psr.ok_value()
        displacement_saddle = (
            dfevent.at["saddle_positions"].copy()
            - dfevent.at["initial_positions"].copy()
        )
        displacement_final = (
            dfevent.at["final_positions"].copy()
            - dfevent.at["initial_positions"].copy()
        )
        sym_matrix = dfevent.at["sym_matrix"][symmetry_index]
        perm_matrix = dfevent.at["sym_perm"][symmetry_index]

        new_displacement_saddle = geometry.transform_positions(
            displacement_saddle, sym_matrix, 0, perm_matrix
        )
        new_displacement_final = geometry.transform_positions(
            displacement_final, sym_matrix, 0, perm_matrix
        )

        saddle_positions = (
            dfevent.at["initial_positions"].copy() + new_displacement_saddle
        )
        final_positions = (
            dfevent.at["initial_positions"].copy() + new_displacement_final
        )

        new_positions_saddle = geometry.transform_positions(
            saddle_positions,
            output_psr.rotation_matrix,
            output_psr.translation_matrix,
            output_psr.permutation_matrix,
        )
        new_positions_final = geometry.transform_positions(
            final_positions,
            output_psr.rotation_matrix,
            output_psr.translation_matrix,
            output_psr.permutation_matrix,
        )
        min2_positions = ase.geometry.wrap_positions(
            new_positions_final, cell=self.system.cell, pbc=True
        )

        current_positions = self.system.positions.copy()
        self.system.update_positions(
            new_positions=new_positions_saddle, atom_idx=neighbors
        )
        if dfevent.at["dE_forward"] > task.e_thr:
            immediate_result = Ok(
                EventRefinementOutput(
                    central_atom_index=at_idx,
                    saddle_positions=self.system.positions.copy(),
                    E_saddle=dfevent["dE_forward"]
                    if self.config.control.active_volume
                    else task.total_energy + dfevent["dE_forward"],
                    num_reference_event=num_reference_event,
                    symmetry_index=symmetry_index,
                    refined="F",
                )
            )
            self.system.update_positions(current_positions)
            return PreparedRefinementTask(
                task=task,
                min2_positions=min2_positions,
                reference_energy_barrier=dfevent["dE_forward"],
                neighbors=neighbors,
                immediate_result=immediate_result,
            )

        if self.config.control.active_volume == True:
            submit_kwargs = {
                "config": self.config,
                "central_atom": at_idx,
                "positions": current_positions.copy(),
                "cell": self.system.cell,
                "types": self.system.types.copy(),
                "saddle_idx": neighbors.copy(),
                "saddle_positions": self.system.positions.copy()[neighbors.copy()],
                "num_reference_event": num_reference_event,
                "symmetry_index": symmetry_index,
            }
        else:
            submit_kwargs = {
                "config": self.config,
                "central_atom": at_idx,
                "positions": self.system.positions.copy(),
                "cell": self.system.cell,
                "types": self.system.types.copy(),
                "saddle_idx": neighbors.copy(),
                "num_reference_event": num_reference_event,
                "symmetry_index": symmetry_index,
            }
        self.system.update_positions(current_positions)
        return PreparedRefinementTask(
            task=task,
            min2_positions=min2_positions,
            reference_energy_barrier=dfevent["dE_forward"],
            neighbors=neighbors,
            submit_kwargs=submit_kwargs,
        )

    def _submit_task(self, prepared: PreparedRefinementTask):
        if prepared.immediate_result is not None:
            future = concurrent.futures.Future()
            future.set_result(prepared.immediate_result)
            return future
        return self.manager.partn_refine(**prepared.submit_kwargs)

    def _finalize_result(
        self,
        res,
        prepared: PreparedRefinementTask,
    ):
        task = prepared.task
        if res.is_ok():
            res.ok_value().min2_positions = prepared.min2_positions
            res.ok_value().num_reference_event = task.num_reference_event
            res.ok_value().symmetry_index = task.symmetry_index
            res.ok_value().saddle_positions = res.ok_value().saddle_positions[
                prepared.neighbors
            ]
            if self.config.control.active_volume == True:
                res.ok_value().dE_forward = res.ok_value().E_saddle
            else:
                res.ok_value().dE_forward = res.ok_value().E_saddle - task.total_energy
            return self.check_refinement_energy(
                res,
                abs(res.ok_value().dE_forward - prepared.reference_energy_barrier),
                self.config.eventsearch.refined_energy_thr,
                task.num_reference_event,
            )

        err = res.err_value()
        if not isinstance(err.variables, dict):
            err.variables = {}
        err.variables["n_ref_event"] = task.num_reference_event
        err.variables.setdefault("num_reference_event", task.num_reference_event)
        err.variables.setdefault("symmetry_index", task.symmetry_index)
        return res

    def check_refinement_energy(
        self,
        result_refine: Result[EventRefinementOutput, ErrorInfo],
        energy_mismatch: float,
        refined_energy_thr: float,
        num_reference_event: int,
    ) -> Result[EventRefinementOutput, ErrorInfo]:
        """Check if the energy barrier of the refinement correspond the one of the reference event.

        Parameters
        ----------
        result_refine : Result[EventRefinementOutput, ErrorInfo]
            Results of the refinement procedure.
        energy_mismatch : float
            Difference between the reference event energy barrier and the refine one.
        refined_energy_thr : float
            maximum allowed difference (in eV) between a reference event's initial barrier energy and its refined barrier energy
        num_reference_event : int
            Reference event this refinement task was attempting, recorded on failure for diagnostics.

        Returns
        -------
        Result[EventRefinementOutput, ErrorInfo]
            list of results of the procedure.

        """
        if energy_mismatch > refined_energy_thr:
            return Err(
                ErrorInfo(
                    type=ErrorType.REFINEMENT_INVALID_ENERGY_BARRIER,
                    message="refinement energy barrier does not match reference one",
                    variables={
                        "n_ref_event": num_reference_event,
                        "num_reference_event": num_reference_event,
                    },
                )
            )
        else:
            return result_refine

    def _get_supposed_ktot(self, tasks: list[RefinementTask]) -> float:
        return sum(task.dfevent.at["k"] for task in tasks)

    def _get_energy_threshold(self, df_reference_events, supposed_ktot):
        tol = self.config.control.refine_thr
        k_thr = supposed_ktot * tol

        # get energy corresponding to the first k value just under k_thr
        mask = df_reference_events["k"] < k_thr
        if mask.any():
            e_value = (
                df_reference_events.loc[mask].sort_values("k").iloc[-1]["dE_forward"]
            )
        else:  # refine no event
            e_value = 0.0
        e_value += 0.1  # to be sure want using condition
        return e_value

    def get_successes_results(self) -> list[EventRefinementOutput]:
        """Return successful results.

        Returns
        -------
        list[EventRefinementOutput]
            list of EventRefinementOutpout dataclass with refine event's informations.

        """
        return [e.ok_value() for e in self.results if e is not None and e.is_ok()]
