"""Module implementing ShapeTable, the persistent table of shape identity.

Distinct from `event_table.ReferenceEventTable` (the table of discovered
transitions between shapes): this is the table of the shapes themselves --
what minima and saddles are known, keyed by their own resolved identity,
independent of how many (if any) transitions currently reference them.
"""

from __future__ import annotations

import pickle
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterator, Literal

import numpy as np

from .parameters import Parameters
from .system import System, Configuration
from .neighbors_list import NeighborsList
from .result import ShapeID, SaddleID, Result, ErrorInfo
from .point_set_registration import Alignment, Cluster, align, cut_cluster
from .utils.io_utils import write_pickle

# How far a shape's searching has got, for keeping the more advanced of two records.
PROGRESS = {"unsearched": 0, "opportunistic": 1, "completed": 2}


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

    Attributes
    ----------
    aliases : dict[str, list[ShapeID | SaddleID]]
        The entries each `topo_id` is also tried against.
    merged : dict[ShapeID, ShapeID]
        Each fused minimum and the minimum it was fused into.
    pending : list[tuple[ShapeID, ShapeID, Cluster]]
        Fusions not yet run: the minimum, the one it goes into, and the cluster both matched.

    """

    def __init__(self, params: Parameters) -> None:
        self.params = params
        self.shape_knowledge: dict[ShapeID, ShapeKnowledge] = {}
        self.saddle_knowledge: dict[SaddleID, SaddleKnowledge] = {}
        self.aliases: dict[str, list[ShapeID | SaddleID]] = {}
        self.merged: dict[ShapeID, ShapeID] = {}
        self.pending: list[tuple[ShapeID, ShapeID, Cluster]] = []
        if params.control.topology_search_status is not None:
            with open(params.control.topology_search_status, "rb") as file:
                data = pickle.load(file)
            self.shape_knowledge = data["shape_knowledge"]
            self.saddle_knowledge = data["saddle_knowledge"]
            # Catalogues written before aliasing and fusion carry none.
            self.aliases = data.get("aliases", {})
            self.merged = data.get("merged", {})
            self.pending = data.get("pending", [])

        self._index: dict[str, dict[str, list]] = {"minimum": {}, "saddle": {}}
        self._indexed = {"minimum": -1, "saddle": -1}
        self._age: dict[ShapeID | SaddleID, int] = {}

    def known_ids(self) -> set[str]:
        """The `id_initial` values with at least one catalogued shape."""
        return {shape.id for shape in self.shape_knowledge}

    def _knowledge(self, kind: Literal["minimum", "saddle"]) -> dict:
        """The catalog of `kind`."""
        return self.shape_knowledge if kind == "minimum" else self.saddle_knowledge

    def _topo_index(self, kind: Literal["minimum", "saddle"]) -> dict[str, list]:
        """`{topo_id: [entries]}` over the `kind` catalog, kept by `resolve` and `retire` and rebuilt when the catalog changed elsewhere."""
        knowledge = self._knowledge(kind)
        if self._indexed[kind] != len(knowledge):
            index: dict[str, list] = {}
            for shape in knowledge:
                index.setdefault(shape.id, []).append(shape)
            self._index[kind], self._indexed[kind] = index, len(knowledge)
        return self._index[kind]

    def canonical(self, shape: ShapeID | SaddleID) -> ShapeID | SaddleID:
        """The minimum `shape` was fused into, or `shape` itself."""
        while shape in self.merged:
            shape = self.merged[shape]
        return shape

    def match(self, shape: ShapeID | SaddleID, cluster: Cluster, thr: float) -> Result[Alignment, ErrorInfo]:
        """Match `shape`'s representative into `cluster` within `thr`."""
        entry = (self.shape_knowledge if isinstance(shape, ShapeID) else self.saddle_knowledge)[shape]
        alignment = align(entry.rep_config, entry.rep_atom_idx, cluster, self.params, thr)
        return alignment

    def _candidates(self, kind: Literal["minimum", "saddle"], topo_id: str, also: list) -> list:
        """The entries `topo_id` is tried against: its own, those it is an alias of, and `also`; an entry with no representative yet is none."""
        key_cls = ShapeID if kind == "minimum" else SaddleID
        knowledge = self._knowledge(kind)
        own = self._topo_index(kind).get(topo_id, [])
        aliased = [shape for shape in self.aliases.get(topo_id, []) if isinstance(shape, key_cls)]
        candidates = [shape for shape in dict.fromkeys(self.canonical(shape) for shape in [*own, *aliased, *also]) if knowledge[shape].rep_config is not None]
        return candidates

    def _alias(self, topo_id: str, shape: ShapeID | SaddleID) -> None:
        """List `shape` under `topo_id`, unless it is minted there or already listed."""
        if shape.id != topo_id and shape not in self.aliases.get(topo_id, []):
            self.aliases[topo_id] = self.aliases.get(topo_id, []) + [shape]

    def resolve(
        self,
        kind: Literal["minimum", "saddle"],
        topo_id: str,
        cluster: Cluster,
        *,
        also: list = (),
        fallback: list = (),
        mint: bool = True,
    ) -> ShapeID | SaddleID | None:
        """Classify `cluster` within the `kind` catalog, minting an entry under `topo_id` when nothing matches.

        Parameters
        ----------
        kind : Literal["minimum", "saddle"]
            The catalog to classify against.
        topo_id : str
            The cluster's topology hash.
        cluster : Cluster
            The geometry, from `cut_cluster`.
        also : list
            Entries also tried.
        fallback : list
            Entries tried only when nothing else matches.
        mint : bool
            Whether to mint when nothing matches.

        Returns
        -------
        ShapeID | SaddleID | None
            The cluster's entry, or `None` when nothing matches and `mint` is off.

        """
        thr = self.params.psr.matching_score_thr
        first = self._candidates(kind, topo_id, also)
        rest = [shape for shape in self._candidates(kind, topo_id, fallback) if shape not in first]
        if any(shape not in self._age for shape in [*first, *rest]):
            # Catalogs keep minting order.
            self._age = {shape: i for i, shape in enumerate([*self.shape_knowledge, *self.saddle_knowledge])}

        # Oldest first. A minimum is tried against every candidate, to find fusions; a saddle is never fused and stops at its first match.
        matches = []
        for candidates in (first, rest):
            for shape in sorted(candidates, key=self._age.__getitem__):
                if self.match(shape, cluster, thr).is_ok():
                    matches += [shape]
                    if kind == "saddle":
                        break
            if matches:
                break

        if matches:
            # The oldest match keeps its ID.
            winner = matches[0]
            self._alias(topo_id, winner)
            if kind == "minimum":
                self.pending += [(shape, winner, cluster) for shape in matches[1:]]
            return winner
        if not mint:
            return None

        # A new entry, represented by the cluster's atoms within rcut.
        positions = cluster.configuration.positions
        radii = np.linalg.norm(positions - positions[cluster.center], axis=1)
        rows = np.flatnonzero(radii <= self.params.atomicenvironment.rcut)

        # A fused minimum's sid still resolves to its survivor.
        key_cls, entry_cls = (ShapeID, ShapeKnowledge) if kind == "minimum" else (SaddleID, SaddleKnowledge)
        index = self._topo_index(kind)
        taken = [shape.sid for shape in [*index.get(topo_id, []), *self.merged] if shape.id == topo_id and isinstance(shape, key_cls)]
        shape = key_cls(topo_id, max(taken) + 1 if taken else 0)
        self._knowledge(kind)[shape] = entry_cls(rep_config=cluster.configuration[rows], rep_atom_idx=int(np.flatnonzero(rows == cluster.center)[0]))
        index.setdefault(topo_id, []).append(shape)
        self._indexed[kind] += 1
        self._age[shape] = len(self._age)
        return shape

    def fusions(self) -> Iterator[tuple[ShapeID, ShapeID]]:
        """Empty the queue, yielding each fusion that still holds: the minimum and the one it goes into."""
        pending, self.pending = self.pending, []
        thr = self.params.psr.matching_score_thr

        for shape, into, cluster in pending:
            # Read after the previous fusion was run.
            shape, into = self.canonical(shape), self.canonical(into)
            if shape == into:
                continue

            # An earlier fusion may have replaced either side by a shape this cluster does not match.
            if self.match(shape, cluster, thr).is_ok() and self.match(into, cluster, thr).is_ok():
                yield shape, into

    def retire(self, shape: ShapeID, into: ShapeID, renamed: dict[int, int]) -> None:
        """Fuse minimum `shape` into `into`: drop its representative, fold its knowledge into `into`'s, and rename the events in `renamed` everywhere.

        Parameters
        ----------
        shape : ShapeID
            The minimum fused.
        into : ShapeID
            The minimum it is fused into.
        renamed : dict[int, int]
            The `idx_ref` each dropped event is now known as.

        """
        self._topo_index("minimum")[shape.id].remove(shape)
        source = self.shape_knowledge.pop(shape)
        self._indexed["minimum"] -= 1
        self.merged[shape] = into

        target = self.shape_knowledge[into]
        target.rediscovery_counts = dict(Counter(target.rediscovery_counts) + Counter(source.rediscovery_counts))
        target.known_rates = {**source.known_rates, **target.known_rates}
        target.known_barriers = {**source.known_barriers, **target.known_barriers}
        target.search_log += source.search_log
        target.escalated = target.escalated or source.escalated
        target.status = max(target.status, source.status, key=PROGRESS.get)
        if target.final_state is None:
            target.final_state, target.final_fraction = source.final_state, source.final_fraction

        for know in self.shape_knowledge.values():
            counts = Counter()
            for idx_ref, count in know.rediscovery_counts.items():
                counts[renamed.get(idx_ref, idx_ref)] += count
            know.rediscovery_counts = dict(counts)
            know.known_rates = {renamed.get(idx_ref, idx_ref): k for idx_ref, k in know.known_rates.items()}
            know.known_barriers = {renamed.get(idx_ref, idx_ref): barriers for idx_ref, barriers in know.known_barriers.items()}
            know.search_log = [SearchRecord(record.outcome, renamed.get(record.idx_ref, record.idx_ref)) for record in know.search_log]

        for topo_id, shapes in self.aliases.items():
            self.aliases[topo_id] = list(dict.fromkeys(into if other == shape else other for other in shapes))
        self._alias(shape.id, into)

    def resolve_live_shape(
        self,
        system: System,
        neighbors_list: NeighborsList,
        atomic_environment,
        atom_idx: int,
        *,
        mint: bool,
    ) -> ShapeID | None:
        """One live atom's `ShapeID`, minted from its own cluster when nothing matches and `mint` is set.

        Parameters
        ----------
        system : System
            The live system.
        neighbors_list : NeighborsList
            The system's neighbour lists.
        atomic_environment : AtomicEnvironment
            The system's environments, giving the atom's `id_initial`.
        atom_idx : int
            The atom.
        mint : bool
            Whether to mint when nothing matches.

        Returns
        -------
        ShapeID | None
            The shape, or `None` when nothing matches and `mint` is off.

        """
        id_initial = atomic_environment.atomic_environment_list[atom_idx]
        if not mint and not self._candidates("minimum", id_initial, []):
            return None

        cluster = cut_cluster(system.configuration, neighbors_list, atom_idx, self.params)
        shape = self.resolve("minimum", id_initial, cluster, mint=mint)
        return shape

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
        data = {key: getattr(self, key) for key in ("shape_knowledge", "saddle_knowledge", "aliases", "merged", "pending")}
        write_pickle(data, self.params.control.topology_search_status_output)
