"""Draw a saved basin and write every state's configuration to extxyz.

Usage: python basin_connectivity.py RUN_DIR/basin_connectivity_N.pickle [...] [--no-show]

Each pickle is read beside its `basin_states_N.pickle`, against its own run
directory, which must hold `input.in` and `reference_table.pickle`. A state's
configuration is, in order:

- `relaxed`: the whole system the basin materialized for it;
- `placed`: the event's final geometry pasted onto its parent, unrelaxed --
  placed again from the reference table through the parent's alignment, for an
  exit the basin never had to identify.

A state's distance is the total distance from the basin's entry state,
sqrt(sum_i |r_i - r_i^entry|^2) over all atoms, minimum image.

Writes beside the pickle `basin_connectivity_N.xyz` (one frame per state,
ascending, distance in each frame's info) and `basin_connectivity_N.svg`, a
spring layout with every node labelled with its distance.
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy
import pandas as pd
from ase import Atoms
from ase.geometry import find_mic
from ase.io import write
from matplotlib.lines import Line2D
from tqdm import tqdm

from pykmc import System
from pykmc.basins import BasinsGenericEvents, StateData
from pykmc.event_table import ReferenceEventTable
from pykmc.parameters import Parameters

ROLES = {
    "entry": ("C0", "basin entry"),
    "explored": ("C2", "basin state, explored"),
    "exit relaxed": ("C1", "exit target, relaxed"),
    "exit placed": ("C3", "exit target, placed only"),
}


def graph_of(df: pd.DataFrame) -> nx.MultiDiGraph:
    """The basin's states and transitions, one edge per connectivity row."""
    graph = nx.MultiDiGraph()
    for idx, row in df.iterrows():
        graph.add_edge(int(row["state"]), int(row["state_connexion"]), key=idx, **row.to_dict())
    return graph


def role_of(graph, state, entry, relaxed) -> str:
    if state == entry:
        return "entry"
    if graph.out_degree(state) > 0:
        return "explored"
    return "exit relaxed" if state in relaxed else "exit placed"


def configurations(graph, saved, run_dir: Path):
    """Per state, `(Configuration, kind, parent)`; states with no recoverable geometry are left out."""
    params = Parameters.from_ini_file(str(run_dir / "input.in"))
    params.control.reference_table = str(run_dir / "reference_table.pickle")
    params.basin.style = "global"  # places the final geometry, not the saddle
    basin = BasinsGenericEvents(params, ReferenceEventTable(params), manager=None)

    configs = {state: (configuration, "relaxed", None) for state, configuration in saved.items()}
    basin.states = {
        state: StateData(system=System.from_configuration(configuration.copy(), pbc=True), environment=None, neighbors_list=None)
        for state, configuration in saved.items()
    }

    failed = []
    pending = [state for state in graph.nodes if state not in configs]
    for state in tqdm(pending, desc="place".ljust(8), ncols=100, ascii=True):
        parents = sorted(p for p in graph.predecessors(state) if p in saved)
        if not parents:
            failed += [(state, "no materialized parent")]
            continue
        parent = parents[0]
        row = next(iter(graph[parent][state].values()))
        result = basin.place_generic_event(parent, int(row["event_connexion"]), int(row["central_atom"]), int(row["sym"]))
        if not result.is_ok():
            failed += [(state, result.err_value())]
            continue
        configs[state] = (result.ok_value().system.configuration, "placed", parent)
        basin.states[parent].release_heavy_objects()

    for state, reason in failed:
        print(f"state {state:5d}: no configuration ({reason})")
    return configs


def distances(configs, entry: int) -> dict[int, float]:
    """Per state, the total minimum-image distance from `entry`'s configuration, in Angstrom."""
    origin = configs[entry][0]
    cell = numpy.asarray(origin.cell)
    distance = {}
    for state, (config, _, _) in configs.items():
        _, lengths = find_mic(config.positions - origin.positions, cell, pbc=True)
        distance[state] = float(numpy.sqrt((lengths**2).sum()))
    return distance


def write_xyz(graph, configs, distance, entry, relaxed, outfile: Path):
    frames = []
    for state in sorted(configs):
        config, kind, parent = configs[state]
        atoms = Atoms(config.types, positions=config.positions, cell=config.cell, pbc=True)
        atoms.info = {
            "state": state, "kind": kind, "role": role_of(graph, state, entry, relaxed),
            "parent": -1 if parent is None else parent, "distance": distance[state],
        }
        frames += [atoms]

    write(outfile, frames, format="extxyz")
    print(f"{outfile}: {len(frames)} of {graph.number_of_nodes()} states")


def draw(graph, distance, entry, relaxed, title: str, outfile: Path):
    """Spring layout of the basin, every node labelled with its distance."""
    layout = nx.spring_layout(graph, seed=0, k=1.5 / numpy.sqrt(graph.number_of_nodes()))
    small = graph.number_of_nodes() <= 60
    side = numpy.clip(numpy.sqrt(graph.number_of_nodes()), 10, 30)

    fig, ax = plt.subplots(figsize=(side, side))
    nx.draw_networkx_edges(graph, layout, ax=ax, edge_color="gray", width=0.6, alpha=0.5, arrows=small)
    for role, (color, _) in ROLES.items():
        nodes = [s for s in graph.nodes if role_of(graph, s, entry, relaxed) == role]
        nx.draw_networkx_nodes(graph, layout, nodelist=nodes, ax=ax, node_color=color, node_size=160 if role == "entry" else (90 if small else 25))

    for state, (x, y) in layout.items():
        d = f"{distance[state]:.2f}" if state in distance else "?"
        if small:
            ax.annotate(f"{state}\n{d} Å", (x, y), xytext=(0, -9), textcoords="offset points", ha="center", va="top", fontsize=7)
        else:
            ax.annotate(f"{state}  {d}", (x, y), xytext=(0, -5), textcoords="offset points", ha="center", va="top", fontsize=4)

    counts = {role: sum(role_of(graph, s, entry, relaxed) == role for s in graph.nodes) for role in ROLES}
    handles = [Line2D([], [], marker="o", linestyle="", markersize=8, color=color, label=f"{label} ({counts[role]})") for role, (color, label) in ROLES.items()]
    ax.legend(handles=handles, loc="upper left", frameon=False)
    ax.set_title(f"{title}: {graph.number_of_nodes()} states, {graph.number_of_edges()} transitions; label: state, total distance from entry state {entry} (Å)")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(outfile)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pickles", nargs="+", type=Path)
    parser.add_argument("--no-show", action="store_true", help="save the figures without opening a window")
    args = parser.parse_args()

    for path in args.pickles:
        graph = graph_of(pd.read_pickle(path))
        with open(path.with_name(path.name.replace("basin_connectivity_", "basin_states_")), "rb") as file:
            saved = pickle.load(file)
        entry, relaxed = saved["entry"], saved["configurations"]

        configs = configurations(graph, relaxed, path.parent)
        distance = distances(configs, entry)
        write_xyz(graph, configs, distance, entry, relaxed, path.with_suffix(".xyz"))
        draw(graph, distance, entry, relaxed, path.stem, path.with_suffix(".svg"))

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
