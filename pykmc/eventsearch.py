"""Module implementing the EventSearch class that deals with the event search procedure."""

import concurrent.futures
import logging
from .result import ErrorInfo, EventSearchOutput, Result, SearchTask
from .system import System
from .enginemanager.lmpi.pool import Manager
from .log import LogKMC
from .utils.geometry import translate
import numpy as np


class EventSearch:
    """Perform event searches and manage results.

    Parameters
    ----------
    system : System
        The atomic system.
    engine : Engine
        The engine used to perform the event search.
    loggers : LogKMC
        The KMC simulation loggers.

    """

    def __init__(
        self, config, system: System, manager: Manager, loggers: LogKMC
    ) -> None:
        self.config = config
        self.system = system
        self.manager = manager
        self.loggers = loggers
        self.results = None
        self.tasks = []

    def execute(self, central_atom_research_list: list[int]) -> None:
        """Execute an event search for each central atom in the central_atom_research_list list.

        It stores the results of the event searches in self.results

        Parameters
        ----------
        central_atom_research_list : list[int]
            list of central atom around which we will perform the event search.

        """
        tasks = self.build_tasks(central_atom_research_list)
        self.loggers.info(
            "log",
            f"\t :=> Searching {len(tasks)} reference events",
        )
        self.tasks = tasks
        self.results = [None] * len(tasks)
        for task_id, result in self._run_tasks(tasks).items():
            self.results[task_id] = result

    def build_tasks(self, central_atom_research_list: list[int]) -> list[SearchTask]:
        """Build event-search tasks with stable ids."""
        return [
            SearchTask(task_id=task_id, central_atom_index=central_atom_index)
            for task_id, central_atom_index in enumerate(central_atom_research_list)
        ]

    def _run_tasks(
        self, tasks: list[SearchTask]
    ) -> dict[int, Result[EventSearchOutput, ErrorInfo]]:
        if not tasks:
            return {}
        if self.config.control.active_volume == True:
            if self.config.activevolume.ract <= self.config.atomicenvironment.rcut:
                raise ValueError(
                    "Active Volume radius is smaller than cutoff radius. Please increase ract or decrease rcut"
                )
            futures = self.manager.partn_search(
                config=self.config,
                central_atom=[task.central_atom_index for task in tasks],
                positions=self.system.positions.copy(),
                cell=self.system.cell.copy(),
                types=self.system.types.copy(),
            )
        else:
            futures = self.manager.partn_search(
                config=self.config,
                central_atom=[task.central_atom_index for task in tasks],
                positions=self.system.positions.copy(),
                cell=self.system.cell.copy(),
                types=self.system.types.copy(),
            )

        future_to_task = {
            future: task for task, future in zip(tasks, futures, strict=False)
        }

        run_results = {}
        for i, future in enumerate(concurrent.futures.as_completed(future_to_task)):
            task = future_to_task[future]
            try:
                result = future.result()
            except Exception as exc:
                self.loggers.error(
                    "log",
                    f"\t task {task.task_id:5d} | atom {task.central_atom_index:6d} | {'RAISE':<5} type={type(exc).__name__}",
                )
                raise
            run_results[task.task_id] = result
            self._log_task_result(task, result)
            self.loggers.progress_bar("progress", i + 1, len(tasks))
        return run_results

    def _log_task_result(
        self, task: SearchTask, result: Result[EventSearchOutput, ErrorInfo]
    ) -> None:
        """Temporary debug logging for per-search outcomes."""
        if not self.loggers.is_enabled_for("log", logging.DEBUG):
            return

        prefix = f"\t task {task.task_id:5d} | atom {task.central_atom_index:6d}"
        if result.is_ok():
            output = result.ok_value()
            self.loggers.debug(
                "log",
                f"{prefix} | {'OK':<5} dE_fwd={output.dE_forward:.4f} eV  dE_bwd={output.dE_backward:.4f} eV  move_atom={output.move_atom_index:6d}",
            )
            return

        error = result.err_value()
        parts = [
            f"type={error.type.name}",
        ]
        if error.details is not None:
            parts.append(f"details={error.details}")
        if error.variables:
            vars_str = ", ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in error.variables.items()
            )
            parts.append(f"variables=({vars_str})")

        self.loggers.debug(
            "log",
            f"{prefix} | {'FAIL':<5} {', '.join(parts)}",
        )

    def retry(self, retry_task_ids: list[int]) -> None:
        """Rerun only the requested event-search tasks."""
        rerun_tasks = [self.tasks[task_id] for task_id in retry_task_ids]
        for task_id, result in self._run_tasks(rerun_tasks).items():
            self.results[task_id] = result
        # for i, at_idx in enumerate(central_atom_research_list):
        #    event_search_output = self.engine.search_event(self.system, at_idx)
        #    self.results.append(event_search_output)
        #    self.loggers.progress_bar(
        #        "progress", i + 1, len(central_atom_research_list)
        #    )

    def _center_event_positions(
        self, event_search_output: EventSearchOutput
    ) -> EventSearchOutput:
        """Translate positions of the events so that the atom that move the most during the event is at the center of the simulation box.

        It is used to that when we store positions in the reference table around the atom that move the most we don't have periodic bound problems.

        Parameters
        ----------
        event_search_output : EventSearchOutput
            The dataclass countaining the event search outputs.

        Returns
        -------
        EventSearchOutput
            The dataclass countaining the event search outputs with translated positions.

        """
        # Translate atoms so that the atom that moves the most is at the center of the cell at start event, prevent pbc problem with psr
        cell = self.system.cell
        ax, ay, az = cell[0][0], cell[1][1], cell[2][2]
        # displacement
        move_atom_idx = event_search_output.move_atom_index
        dx, dy, dz = (
            ax / 2 - event_search_output.min1_positions[move_atom_idx][0],
            ay / 2 - event_search_output.min1_positions[move_atom_idx][1],
            az / 2 - event_search_output.min1_positions[move_atom_idx][2],
        )
        displacement = np.array([dx, dy, dz])
        event_search_output.min1_positions = translate(
            event_search_output.min1_positions, displacement, cell
        )
        event_search_output.saddle_positions = translate(
            event_search_output.saddle_positions, displacement, cell
        )
        event_search_output.min2_positions = translate(
            event_search_output.min2_positions, displacement, cell
        )
        event_search_output.cell = cell
        return event_search_output

    def get_successes_results(self) -> list[EventSearchOutput]:
        """Return a list of only successful event searches.

        Returns
        -------
        list[EventSearchOutput]
            List of successful event searches.

        """
        return [
            self._center_event_positions(e.ok_value())
            for e in self.results
            if e is not None and e.is_ok()
        ]
