"""Manages Point Set Registration (shape matching) methods."""

from dataclasses import dataclass

from .utils.io_utils import capture_output
import ira_mod
import numpy as np
from .result import Result, ErrorInfo, PSROutput, Ok, Err, ErrorType
from .parameters import Parameters
from .system import System, Configuration
import pandas as pd
from .neighbors_list import NeighborsList
from .utils.geometry import unwrap_around, wrap_configuration


class PointSetRegistration:
    """Perform a point set registration between a reference event and an atomic environment of an atom based on the configuration parameters.

    Parameters
    ----------
    params : Parameters
        The configuration.
    system : System
        The atomic system.
    dfevent : pd.Series
        The reference event.
    neighbors_list : NeighborsList
        The NeighborsList of the System.
    central_atom_index : int
        Index of the central atom in the System for which we want to perfrom the point set registration.

    """

    def __init__(
        self,
        params: Parameters,
        system: System,
        dfevent: pd.Series,
        neighbors_list: NeighborsList,
        central_atom_index: int,
    ) -> None:
        self.system = system
        self.params = params
        self.dfevent = dfevent
        self.neighbors_list = neighbors_list
        self.central_atom_index = central_atom_index
        self.psr_style = self.params.psr.style

    def match(self, thr: float) -> Result[Alignment, ErrorInfo]:
        """Match the event's initial geometry into the central atom's cluster within `thr`, with the configured style."""
        match self.psr_style:
            case "ira":
                return self.ira(self.central_atom_index, thr)
            case _:
                raise Exception("Point set registration style unknown")

    def ira(self, central_atom_index: int, thr: float) -> Result[Alignment, ErrorInfo]:
        """Match the event's initial geometry into `central_atom_index`'s cluster with IRA, within `thr`."""
        cluster = cut_cluster(self.system.configuration, self.neighbors_list, central_atom_index, self.params)
        alignment = align(self.dfevent.at["initial_configuration"], int(self.dfevent.at["move_atom_idx"]), cluster, self.params, thr)
        return alignment


@dataclass
class Cluster:
    """One atom's neighbourhood as matched against a catalogued geometry.

    Attributes
    ----------
    configuration : Configuration
        The atoms within `rcut` + `accept_thr` of the centre, unwrapped around it.
    atoms : list[int]
        The system atom of each row.
    center : int
        The row of the centre.

    """

    configuration: Configuration
    atoms: list[int]
    center: int


@dataclass
class Alignment:
    """A catalogued geometry matched into a `Cluster`.

    Attributes
    ----------
    cluster : Cluster
        The cluster matched into.
    psr : PSROutput
        The transform carrying `cluster` onto the catalogued geometry.
    partners : np.ndarray of int
        The row of `cluster` matched to each row of the catalogued geometry.

    """

    cluster: Cluster
    psr: PSROutput
    partners: np.ndarray

    def place(self, configuration: Configuration, atoms: list[int]) -> Configuration:
        """`configuration`, in the catalogued geometry's frame and rows, carried onto the system atoms `atoms`; an atom with no partner keeps its place."""
        positions = self.cluster.configuration.positions.copy()
        positions[self.partners] = (configuration.positions - self.psr.translation_matrix) @ self.psr.rotation_matrix

        rows = {atom: row for row, atom in enumerate(self.cluster.atoms)}
        selected = [rows[atom] for atom in atoms]
        placed = wrap_configuration(Configuration(types=np.asarray(self.cluster.configuration.types)[selected], positions=positions[selected], cell=self.cluster.configuration.cell))
        return placed


def cut_cluster(configuration: Configuration, neighbors_list: NeighborsList, center_atom: int, params: Parameters) -> Cluster:
    """`center_atom`'s atoms within `rcut` + `accept_thr` in `configuration`, unwrapped around it."""
    atoms = neighbors_list.ball(center_atom, params.atomicenvironment.rcut + params.psr.accept_thr)
    cluster = Cluster(configuration=unwrap_around(configuration[atoms], configuration.positions[center_atom]), atoms=atoms, center=atoms.index(center_atom))
    return cluster


def align(reference: Configuration, reference_center: int, cluster: Cluster, params: Parameters, thr: float) -> Result[Alignment, ErrorInfo]:
    """Match `reference` into `cluster` within `thr`, leaving no cluster atom closer than `rcut` - `accept_thr` to its centre unmatched.

    Parameters
    ----------
    reference : Configuration
        The catalogued geometry, cut at `rcut`.
    reference_center : int
        The row of `reference` holding its centre.
    cluster : Cluster
        The geometry matched into.
    params : Parameters
        The run's configuration.
    thr : float
        The largest score accepted.

    Returns
    -------
    Result[Alignment, ErrorInfo]
        The match, or why there is none.

    """
    full = params.atomicenvironment.coloring_mode == "full"
    result = check_match(simple_ira(reference, cluster.configuration, params.ira.kmax_factor, full=full, candidate1=reference_center, candidate2=cluster.center), thr)
    if not result.is_ok():
        return result

    psr = result.ok_value()
    partners = psr.permutation_matrix[: len(reference)]
    unmatched = psr.permutation_matrix[len(reference):]
    radii = np.linalg.norm(cluster.configuration.positions[unmatched] - cluster.configuration.positions[cluster.center], axis=1)
    inner = params.atomicenvironment.rcut - params.psr.accept_thr
    if (radii < inner).any():
        return Err(ErrorInfo(type=ErrorType.PSR_ATOM_COUNT_MISMATCH, message="Cluster atom inside rcut left unmatched", details=f"unmatched atom at {radii.min():.3f} A, inside {inner:.3f} A"))
    return Ok(Alignment(cluster=cluster, psr=psr, partners=partners))


def check_match(
    result_match: Result[PSROutput, ErrorInfo], matching_score: float
) -> Result[PSROutput, ErrorInfo]:
    """Check if a result from the point set registration method is valid and gives a matching score lower than the matching score threshold defined in the configuration.

    Parameters
    ----------
    result_match : Result[PSROutput, ErrorInfo]
        Result of the PSR procedure.
    matching_score : float
        matching score threshold.

    Returns
    -------
    Result[PSROutput, ErrorInfo]
        Result of the check.

    """
    if not result_match.is_ok():
        return result_match  # ErrorInfo no match
    else:
        if result_match.ok_value().matching_score > matching_score:
            return Err(
                ErrorInfo(
                    type=ErrorType.PSR_MATCHING_SCORE_ABOVE_ACCEPTANCE_THRESHOLD,
                    message="PSR found a match but matching score is above acceptance threshold",
                    details="Hausdorff distance = {}, acceptance threshold = {} ".format(
                        result_match.ok_value().matching_score, matching_score
                    ),
                    variables={
                        "matching_score": result_match.ok_value().matching_score
                    },
                )
            )

        else:
            return result_match  # Ok(PSROutput)


_ira = None
def _get_ira() -> "ira_mod.IRA":
    """One shared `ira_mod.IRA()` instance, reused across every `simple_ira` call."""
    global _ira
    if _ira is None: _ira = ira_mod.IRA()
    return _ira

@capture_output()
def simple_ira(
    configuration_1: Configuration,
    configuration_2: Configuration,
    kmax_factor: float,
    full: bool = False,
    candidate1: int | None = None,
    candidate2: int | None = None,
) -> Result[PSROutput, ErrorInfo]:
    """Match `configuration_1` into `configuration_2` with IRA, scoring `configuration_1`'s atoms only; `full` gates real vs. dummy ("X") types.

    `configuration_2` may hold more atoms, left unmatched past `len(configuration_1)` in the permutation.
    `candidate1`/`candidate2`, if given, seed the search with a known
    matching atom pair (e.g. each configuration's central atom).
    """
    nat1 = len(configuration_1)
    nat2 = len(configuration_2)
    if nat1 > nat2:
        return Err(ErrorInfo(type=ErrorType.PSR_ATOM_COUNT_MISMATCH, message="First configuration larger than the second", details=f"{nat1} atoms into {nat2}"))
    typ1 = list(configuration_1.types) if full else nat1 * ["X"]
    typ2 = list(configuration_2.types) if full else nat2 * ["X"]
    candidate_kwargs = {"candidate1": candidate1, "candidate2": candidate2} if candidate1 is not None else {}
    ira = _get_ira()
    try:
        rmat, tr, perm, dh = ira.match(
            nat1, typ1, configuration_1.positions, nat2, typ2, configuration_2.positions, kmax_factor, **candidate_kwargs
        )

        return Ok(
            PSROutput(
                rotation_matrix=rmat,
                translation_matrix=tr,
                permutation_matrix=perm,
                matching_score=dh,
            )
        )
    except Exception:
        return Err(
            ErrorInfo(
                type=ErrorType.PSR_NO_MATCH_FOUND,
                message="IRA did not find a match",
            )
        )

