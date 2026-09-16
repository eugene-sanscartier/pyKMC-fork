"""Module implementing ShapeTable, the persistent table of shape identity.

Distinct from `event_table.ReferenceEventTable` (the table of discovered
transitions between shapes): this is the table of the shapes themselves --
what minima and saddles are known, keyed by their own resolved identity,
independent of how many (if any) transitions currently reference them.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Literal

from .parameters import Parameters
from .system import System, Configuration
from .neighbors_list import NeighborsList
from .result import ShapeID, SaddleID
from .point_set_registration import simple_ira, check_match
from .utils.geometry import unwrap_around


@dataclass
class SearchRecord:
    """One dispatched search's outcome against a shape, in the order it happened.

    Attributes
    ----------
    outcome : Literal["new", "duplicate", "error"]
        "new": a genuinely new event was catalogued for this shape for the
        first time. "duplicate": the search rematched an event this shape
        already knew about. "error": the search produced no usable evidence
        at all (no saddle found, runtime error, ambiguous minima, energy/
        asymmetry rejection, ...) -- the specific reason is not
        distinguished here.
    idx_ref : int | None
        The (re)discovered event, set only for "new"/"duplicate".

    """

    outcome: Literal["new", "duplicate", "error"]
    idx_ref: int | None = None


@dataclass
class ShapeKnowledge:
    """Cross-step knowledge for one `ShapeID`, persisted on `ShapeTable`.

    Survives across steps and restarts -- it is the record of "what do we
    know, ever, about this shape's event population," independent of
    whether any of that knowledge came from a dispatched search of this
    exact shape or an opportunistic discovery made while searching
    something else. Keyed externally (on `ShapeTable.shape_knowledge`) by
    the `ShapeID` itself, not by its coarse id alone -- nauty's coarse id
    can collide across genuinely different local neighborhoods, so a
    coarse-only key would let one shape's convergence history mask a
    different shape's total lack of evidence just because they happen to
    share a coarse id.

    Attributes
    ----------
    status : Literal["unsearched", "opportunistic", "completed"]
        "unsearched": no rows and no dispatched session yet. "opportunistic":
        has rediscovery/rate data (e.g. from a disconnected event landing on
        this shape) but has never been through its own dispatched session.
        "completed": a dispatched session (adaptive convergence/cap, or a
        full `nsearch` round in non-adaptive mode) for this exact shape has
        finished -- the only status `resolve_new_shapes()` trusts as
        "don't search again."
    rediscovery_counts : dict[int, int]
        idx_ref -> number of times catalogued, same semantics as
        `pykmc.adaptive_search.ShapeSearchStats.rediscovery_counts` but
        accumulated over this shape's whole lifetime, not just one step's
        session.
    known_rates : dict[int, float]
        idx_ref -> cached rate constant k_i, same lifetime as above.
    known_barriers : dict[int, tuple[float, float]]
        idx_ref -> (dE_forward, dE_backward) at first (re)discovery, same
        lifetime as `rediscovery_counts`.
    search_log : list[SearchRecord]
        Every dispatched search that ever touched this shape (its own
        dispatched session, or another shape's search whose outcome
        resolved to this shape), in the real chronological order it
        happened -- unlike `rediscovery_counts`/`known_barriers`, which only
        keep running totals, this is a full ordered replay of what
        happened, including searches that produced no usable evidence.
    escalated : bool
        Whether this shape's own dispatched adaptive session was ever
        flagged escalated (see `pykmc.adaptive_search.ShapeSearchStats.escalated`)
        before it finished. Always `False` for a shape only ever searched in
        non-adaptive (`nsearch`) mode, which has no escalation concept.
    final_state : Literal["converged", "capped"] | None
        This shape's own dispatched adaptive session's terminal outcome, if
        it ever had one. `None` for a shape only ever searched in
        non-adaptive mode.
    final_fraction : float | None
        This shape's adaptive session's last computed undiscovered-mass
        fraction (see `pykmc.adaptive_search.ShapeSearchStats.last_fraction`),
        if it ever had one. `None` for a shape only ever searched in
        non-adaptive mode.
    rep_config : Configuration | None
        This shape's own representative local-cluster geometry, set once
        when the shape is first resolved (minted) and never changed
        afterward -- `ShapeTable` is the sole catalog of minima identity, so
        this is the only geometry any later classification or resolution
        compares a candidate against. `None` only for the brief span before
        a shape has ever actually been resolved (e.g. a `setdefault`-created
        placeholder from `mark_shape_completed`).
    rep_atom_idx : int | None
        `rep_config`'s own central atom index, `None` iff `rep_config` is
        `None`.

    """

    status: Literal["unsearched", "opportunistic", "completed"] = "unsearched"
    rediscovery_counts: dict[int, int] = field(default_factory=dict)
    known_rates: dict[int, float] = field(default_factory=dict)
    known_barriers: dict[int, tuple[float, float]] = field(default_factory=dict)
    search_log: list[SearchRecord] = field(default_factory=list)
    escalated: bool = False
    final_state: Literal["converged", "capped"] | None = None
    final_fraction: float | None = None
    rep_config: Configuration | None = None
    rep_atom_idx: int | None = None

    def record(
        self,
        idx_ref: int,
        k: float,
        dE_forward: float,
        dE_backward: float,
        outcome: Literal["new", "duplicate"],
    ) -> None:
        """Record one sighting of `idx_ref`, rate `k`, barriers, and log it to `search_log`.

        Advances `status` from "unsearched" to "opportunistic" if this is the
        first knowledge ever recorded for this shape; never downgrades a
        "completed" shape back to "opportunistic".
        """
        self.rediscovery_counts[idx_ref] = self.rediscovery_counts.get(idx_ref, 0) + 1
        self.known_rates[idx_ref] = k
        self.known_barriers[idx_ref] = (dE_forward, dE_backward)
        self.search_log += [SearchRecord(outcome, idx_ref)]
        if self.status == "unsearched":
            self.status = "opportunistic"

    def record_failure(self) -> None:
        """Log a dispatched search against this shape that produced no usable evidence."""
        self.search_log += [SearchRecord("error")]


@dataclass
class SaddleKnowledge:
    """A saddle's permanent catalog entry: its own representative geometry, nothing else.

    Kept separate from `ShapeKnowledge` (minima) purely so a candidate's
    matching pool never mixes the two kinds -- a saddle is never
    independently dispatched or searched the way a minimum is (it only ever
    arises as a byproduct of a minimum's dispatched search), so it carries
    none of `ShapeKnowledge`'s search-completeness/rediscovery bookkeeping.

    Attributes
    ----------
    rep_config : Configuration
        This saddle's own representative local-cluster geometry, set once
        when it is first resolved (minted) and never changed afterward.
    rep_atom_idx : int
        `rep_config`'s own central atom index.

    """

    rep_config: Configuration
    rep_atom_idx: int


class ShapeTable:
    """The persistent table of shape identity: what minima and saddles are known.

    Distinct from `event_table.ReferenceEventTable` (the table of discovered
    transitions between shapes): a shape's existence here is independent of
    how many (if any) transitions in `ReferenceEventTable.table` currently
    reference it -- a shape resolved live with zero catalogued events is
    just as real an entry as one with many.

    Minima (`shape_knowledge`) and saddles (`saddle_knowledge`) are kept in
    separate dicts, keyed by `ShapeID`/`SaddleID` respectively, so resolving
    one never has to filter out entries of the other kind. Each entry holds
    exactly one permanent representative geometry, set once when the shape
    is first resolved (minted) and never replaced afterward.

    Parameters
    ----------
    params : Parameters
        The atomic simulations configuration.

    """

    def __init__(self, params: Parameters) -> None:
        self.params = params
        if params.control.topology_search_status is not None:
            with open(params.control.topology_search_status, "rb") as file:
                data = pickle.load(file)
            self.shape_knowledge: dict[ShapeID, ShapeKnowledge] = data["shape_knowledge"]
            self.saddle_knowledge: dict[SaddleID, SaddleKnowledge] = data["saddle_knowledge"]
        else:
            self.shape_knowledge: dict[ShapeID, ShapeKnowledge] = {}
            self.saddle_knowledge: dict[SaddleID, SaddleKnowledge] = {}
        self._shape_by_id: dict[str, list] = {}
        self._shape_indexed = 0
        self._saddle_by_id: dict[str, list] = {}
        self._saddle_indexed = 0

    def known_ids(self) -> set[str]:
        """The `id_initial` values with at least one catalogued shape."""
        return {shape.id for shape in self.shape_knowledge}

    @staticmethod
    def index_by_id(knowledge: dict) -> dict:
        """Precompute {topo_id: sorted [shapes sharing it]} over `knowledge`, to pass into
        `_classify`'s `by_id` -- skips its default O(n) per-call scan of the whole dict. Meant for a
        caller's own frozen snapshot (e.g. reconciling many candidates against one fixed prior-state
        copy) that `self.shape_knowledge`/`self.saddle_knowledge` themselves don't need: `_classify`/
        `_resolve` already maintain and reuse their own live index for those two (`_live_index`)."""
        index: dict = {}
        for shape in sorted(knowledge):
            index.setdefault(shape.id, []).append(shape)
        return index

    def _live_index(self, knowledge: dict) -> dict | None:
        """The persistent, incrementally-maintained {topo_id: [shapes], ascending sid} index behind
        `self.shape_knowledge`/`self.saddle_knowledge` -- `None` for any other dict (a caller's own
        snapshot has no persistent index to offer; use `index_by_id` for those instead).

        Both dicts only ever grow by appending a new key (`_resolve`'s mint, or the handful of
        `setdefault` calls elsewhere in this class) -- never replaced or removed -- and every mint
        assigns `max(existing sid for this topo_id) + 1`, so entries for one topo_id are already in
        ascending-sid order as inserted. Catching up after new entries just means indexing whatever's
        been appended since the last call, not rebuilding from scratch: turns what would otherwise be
        an O(n) rescan of the whole table on every single classification (profiled: the dominant cost
        of reconciling shapes across independent runs, where a live run's own `self.shape_knowledge`
        can hold tens of thousands of entries by the time it's read back) into one O(1) amortized
        lookup per call."""
        if knowledge is self.shape_knowledge: by_id, attr = self._shape_by_id, "_shape_indexed"
        elif knowledge is self.saddle_knowledge: by_id, attr = self._saddle_by_id, "_saddle_indexed"
        else: return None

        items = list(knowledge.items())
        for shape, _ in items[getattr(self, attr):]: by_id.setdefault(shape.id, []).append(shape)
        setattr(self, attr, len(items))
        return by_id

    def _classify(
        self,
        knowledge: dict,
        topo_id: str,
        configuration: Configuration,
        move_atom_idx: int | None = None,
        by_id: dict | None = None,
    ) -> int | None:
        """Match `configuration` against every representative in `knowledge` sharing `topo_id`; `None` if none match.

        Shared mechanism behind minima/saddle classification, live or
        cataloguing-time: `knowledge` (`self.shape_knowledge` or
        `self.saddle_knowledge`) already holds exactly one representative
        geometry per entry, so this is a single IRA/PSR comparison per
        candidate sid, tried in ascending sid order for determinism.

        Parameters
        ----------
        knowledge : dict[ShapeID, ShapeKnowledge] | dict[SaddleID, SaddleKnowledge]
            The catalog to classify against.
        topo_id : str
            The candidate's coarse topology hash (``id_initial`` or ``id_saddle``).
        configuration : Configuration
            The candidate's local-cluster types/positions/cell.
        move_atom_idx : int | None
            Index of the candidate's own central/moving atom within ``configuration``, if known -- seeds the IRA search against each representative's own ``rep_atom_idx`` instead of leaving it to find a geometric center.
        by_id : dict | None
            Optional `index_by_id(knowledge)` result -- when given, replaces the default O(n) scan
            of `knowledge` with an O(1) lookup. Only pass this for a `knowledge` frozen for the whole
            span the index is reused over; omit it (the default) for anything still being mutated.

        Returns
        -------
        int | None
            The matching sid, or `None` if nothing under `topo_id` matches
            (including the case where nothing is catalogued under it at all).

        """
        full = self.params.atomicenvironment.coloring_mode == "full"
        if by_id is None: by_id = self._live_index(knowledge)
        candidates = by_id.get(topo_id, []) if by_id is not None else sorted(shape for shape in knowledge if shape.id == topo_id)
        for shape in candidates:
            know = knowledge[shape]
            result = simple_ira(
                configuration,
                know.rep_config,
                self.params.ira.kmax_factor,
                full=full,
                candidate1=move_atom_idx,
                candidate2=know.rep_atom_idx,
            )
            if check_match(result, self.params.psr.matching_score_thr).is_ok():
                return shape.sid
        return None

    def _resolve(
        self,
        knowledge: dict,
        key_cls,
        entry_cls,
        topo_id: str,
        configuration: Configuration,
        move_atom_idx: int | None,
    ) -> int:
        """Classify-or-mint: match `configuration` against `knowledge` under `topo_id`, minting a fresh entry if nothing matches.

        Shared mechanism behind `resolve_sid` (minima or saddle -- `key_cls`/
        `entry_cls` select which) and `resolve_live_shape`'s mint path.
        """
        by_id = self._live_index(knowledge)
        sid = self._classify(knowledge, topo_id, configuration, move_atom_idx, by_id=by_id)
        if sid is not None:
            return sid

        existing = [shape.sid for shape in by_id.get(topo_id, [])] if by_id is not None else [shape.sid for shape in knowledge if shape.id == topo_id]
        new_sid = max(existing) + 1 if existing else 0
        knowledge[key_cls(topo_id, new_sid)] = entry_cls(
            rep_config=configuration, rep_atom_idx=move_atom_idx
        )
        return new_sid

    def resolve_sid(
        self,
        kind: Literal["initial", "saddle"],
        topo_id: str,
        configuration: Configuration,
        move_atom_idx: int | None = None,
    ) -> int:
        """Resolve a candidate geometry's sid under `topo_id`, within the `kind` catalog -- together they form its `ShapeID`/`SaddleID`.

        `kind="initial"` classifies against `self.shape_knowledge` (minima).
        `kind="saddle"` classifies against `self.saddle_knowledge`,
        free-standing -- the same transition-state geometry is recognized as
        one catalogued saddle regardless of which reaction pathway reaches
        it, unlike the old pathway-scoped resolution this replaced. Either
        way, if nothing matches, this candidate becomes a freshly minted
        entry's own permanent representative.
        """
        if kind == "initial":
            knowledge, key_cls, entry_cls = self.shape_knowledge, ShapeID, ShapeKnowledge
        else:
            knowledge, key_cls, entry_cls = self.saddle_knowledge, SaddleID, SaddleKnowledge
        return self._resolve(knowledge, key_cls, entry_cls, topo_id, configuration, move_atom_idx)

    def _live_configuration(
        self, system: System, neighbors_list: NeighborsList, atom_idx: int
    ) -> tuple[Configuration, int]:
        """This atom's own rcut-neighbor cluster, unwrapped around itself, and its own index within it."""
        nl = neighbors_list.get_neighbors("rcut", atom_idx).copy()
        cfg = unwrap_around(system.configuration[nl], system.positions[atom_idx])
        return cfg, nl.index(atom_idx)

    def classify_live_sid(
        self,
        system: System,
        neighbors_list: NeighborsList,
        central_atom_index: int,
        id_initial: str,
    ) -> int | None:
        """Classify a live atom's ``sid_initial`` against already-catalogued shapes sharing its ``id_initial``.

        Unlike `resolve_sid` (used when cataloguing a freshly found event,
        which always assigns a new sid if none match), this never invents
        one: it only reports whether the atom's current geometry already
        matches a known `ShapeID`, so callers can decide whether a fresh
        event search is warranted for it.

        Parameters
        ----------
        system : System
            The live atomic system.
        neighbors_list : NeighborsList
            The system's neighbor lists.
        central_atom_index : int
            Index of the atom to classify.
        id_initial : str
            The atom's coarse topology hash.

        Returns
        -------
        int | None
            The matching ``sid_initial``, or ``None`` if the atom's live
            geometry matches no `ShapeID` already catalogued under
            ``id_initial`` (including the case where ``id_initial`` has no
            rows at all).

        """
        if not any(shape.id == id_initial for shape in self.shape_knowledge):
            return None

        live_configuration, move_atom_idx = self._live_configuration(
            system, neighbors_list, central_atom_index
        )
        return self._classify(self.shape_knowledge, id_initial, live_configuration, move_atom_idx)

    def resolve_live_shape(
        self,
        system: System,
        neighbors_list: NeighborsList,
        atomic_environment,
        atom_idx: int,
        *,
        mint: bool,
    ) -> ShapeID | None:
        """Resolve one atom's live `ShapeID`.

        With `mint=False`, never invents one: `None` if the atom's current
        geometry matches nothing already catalogued under its `id_initial`,
        so a caller can decide whether a fresh search is warranted at all.

        With `mint=True`, always returns a real `ShapeID`, minting one backed
        by the atom's own geometry if nothing matches -- meant to be called
        once a search has already been dispatched for this atom: by then
        there's nothing left to decide, only a shape left to record
        knowledge against, event or no event. The minted representative is
        this atom's own rcut-neighbor cluster, unwrapped around itself, not
        anything the dispatched search returns, so this never has to wait on
        (or depend on the outcome of) that search.
        """
        id_initial = atomic_environment.atomic_environment_list[atom_idx]
        if not mint:
            sid = self.classify_live_sid(system, neighbors_list, atom_idx, id_initial)
            return None if sid is None else ShapeID(id_initial, sid)

        live_configuration, move_atom_idx = self._live_configuration(system, neighbors_list, atom_idx)
        sid = self.resolve_sid("initial", id_initial, live_configuration, move_atom_idx)
        return ShapeID(id_initial, sid)

    def record_shape_knowledge(
        self,
        id_initial: str,
        sid_initial: int,
        idx_ref: int,
        k: float,
        dE_forward: float,
        dE_backward: float,
        outcome: Literal["new", "duplicate"],
    ) -> None:
        """Record one sighting (new catalogue entry or rediscovery) of `idx_ref` under the shape `ShapeID(id_initial, sid_initial)`.

        Called unconditionally for every successful catalogue outcome
        (`ReferenceEventTable.add_events`), regardless of which atom's
        search produced it or whether this shape has an open per-step
        adaptive session -- this is the single place persistent, cross-step
        knowledge is kept up to date.
        """
        know = self.shape_knowledge.setdefault(
            ShapeID(id_initial, sid_initial), ShapeKnowledge()
        )
        know.record(idx_ref, k, dE_forward, dE_backward, outcome)

    def record_search_failure(self, shape: ShapeID) -> None:
        """Log one dispatched search against `shape` that produced no usable evidence.

        Covers every reason a search can come up empty (no saddle found,
        runtime error, ambiguous minima, energy/asymmetry rejection, ...)
        without distinguishing which -- see `SearchRecord`.
        """
        self.shape_knowledge.setdefault(shape, ShapeKnowledge()).record_failure()

    def record_adaptive_outcome(
        self,
        shape: ShapeID,
        *,
        state: Literal["converged", "capped"],
        escalated: bool,
        final_fraction: float | None,
    ) -> None:
        """Persist `shape`'s adaptive-session terminal telemetry: its final state, whether it was ever escalated, and its last undiscovered-mass fraction.

        Called only from the adaptive dispatch path
        (`AdaptiveSearchSession.finalize`), alongside `mark_shape_completed`
        -- a shape only ever searched in non-adaptive (`nsearch`) mode never
        gets this called, so these fields stay at their defaults for it.
        """
        know = self.shape_knowledge.setdefault(shape, ShapeKnowledge())
        know.final_state = state
        know.escalated = escalated
        know.final_fraction = final_fraction

    def mark_shape_completed(self, shape: ShapeID) -> None:
        """Mark `shape`'s persistent search-completeness status `"completed"`.

        Together with `record_shape_knowledge`/`get_shape_knowledge`/
        `forget`, the only sanctioned way to touch `shape_knowledge` --
        callers must never read or write that dict directly.
        """
        self.shape_knowledge.setdefault(shape, ShapeKnowledge()).status = "completed"

    def get_shape_knowledge(self, shape: ShapeID) -> ShapeKnowledge | None:
        """Look up `shape`'s persistent knowledge, or `None` if never sighted."""
        return self.shape_knowledge.get(shape)

    def needs_search(self, shape: ShapeID) -> bool:
        """Whether `shape`'s own dispatched search round has yet to finish.

        The one definition of "we don't know this shape's reactions yet",
        shared by the search trigger (`KMC.resolve_new_shapes`) and basin
        exploration's explorability gate
        (`BasinsGenericEvents.unknown_environment`). The two must agree, or
        the basin builds connectivity for an atom the main loop still
        considers unsearched. A finished search that found nothing still
        counts as done: some shapes genuinely have no events to find, and
        re-searching them forever would never terminate.
        """
        know = self.shape_knowledge.get(shape)
        return know is None or know.status != "completed"

    def forget(self, shape: ShapeID, idx_ref: int) -> None:
        """Prune `idx_ref` out of `shape`'s rediscovery bookkeeping, if `shape` is catalogued.

        Called by `ReferenceEventTable.remove()` for each row it deletes, so
        a removed transition's `idx_ref` doesn't keep skewing the
        Good-Turing undiscovered-mass estimate or `n_distinct_events` stat.
        `shape`'s own catalog entry is never deleted here -- a shape with
        zero remaining table rows is a normal state, not staleness.
        """
        know = self.shape_knowledge.get(shape)
        if know is not None:
            know.rediscovery_counts.pop(idx_ref, None)
            know.known_rates.pop(idx_ref, None)
            know.known_barriers.pop(idx_ref, None)

    def save(self) -> None:
        """Save the shape catalogs to `params.control.topology_search_status_output`, if set."""
        if self.params.control.topology_search_status_output is None:
            return
        with open(self.params.control.topology_search_status_output, "wb") as file:
            pickle.dump(
                {"shape_knowledge": self.shape_knowledge, "saddle_knowledge": self.saddle_knowledge},
                file,
            )
