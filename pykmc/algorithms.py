"""Provide functions for different Kinetic Monte Carlo (KMC) algorithms."""

import random
import numpy as np
import math as m


def rank_by_coverage(items: list, rates: list[float] | np.ndarray, coverage: float) -> tuple[list, int]:
    """Rank `items` by descending rate, and size the leading group reaching `coverage` of the total.

    Parameters
    ----------
    items : list
        What is being ranked, one per entry of `rates`.
    rates : list of float or np.ndarray of float
        Each item's rate constant, in any order.
    coverage : float
        Fraction of the total rate the leading group must reach. At 1 or
        above the group is every rate, including those too small to move a
        float64 running sum.

    Returns
    -------
    tuple[list, int]
        - ranked_items : `items` by descending rate.
        - n : how many of them the leading group holds; at least one unless
          `items` is empty.

    """
    k = np.asarray(rates, dtype=float)
    order = np.argsort(-k, kind="stable")
    ranked_items = [items[i] for i in order]
    if len(k) == 0 or coverage >= 1.0:
        return ranked_items, len(k)

    k_cumulative = np.cumsum(k[order])
    n = int(np.searchsorted(k_cumulative, coverage * k_cumulative[-1], side="left")) + 1
    return ranked_items, n


def rejection_free(l_k: list[float] | np.ndarray) -> tuple[int, float]:
    """Select an event index and calculates time step using the rejection-free KMC algorithm.

    Parameters
    ----------
    l_k : list of float or np.ndarray of float
        List or array of individual event rate constants

    Returns
    -------
    tuple[int, float]
        - idx_selected_event : int
            The index of of l_k of the selected event.
        - delta_t : float
            The time step update

    """
    # compute cumulative rate constant
    k_cumulative = [np.sum(l_k[:i]) for i in range(1, len(l_k) + 1)]
    rand = random.random()
    # find event index satisfy ki-1<rand1ktot<ki
    idx_selected_event = np.searchsorted(
        k_cumulative, rand * k_cumulative[-1], side="left"
    )

    # compute associated update time:
    delta_t = -m.log(random.random()) / k_cumulative[-1]
    return idx_selected_event, delta_t, k_cumulative[-1]
