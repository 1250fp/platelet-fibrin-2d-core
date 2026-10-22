"""Permanent zero-at-birth fiber--fiber cohesion with explicit valence."""

from __future__ import annotations

import numpy as np

from .config import Program19Config
from .spatial import NeighborIndex
from .state import SimulationState


def cohesion_degrees(state: SimulationState) -> np.ndarray:
    """Number of cohesion links attached to each node."""
    degree = np.zeros(len(state.nodes_um), dtype=int)
    for a, b in state.cohesion_pairs:
        degree[a] += 1
        degree[b] += 1
    return degree


def add_new_cohesion_links(state: SimulationState, config: Program19Config) -> int:
    """Create new permanent fibre--fibre cohesion links; returns how many were made.

    Candidate node pairs closer than ``cohesion_distance_um`` are linked unless
    they are already bonded, lie on the same fibre (when
    ``cohesion_exclude_same_fiber``), or would exceed the valence limits
    ``cohesion_max_total_neighbors`` / ``cohesion_max_new_per_node_per_search``.
    The rest length is the distance at formation, so a new link starts force-free
    and afterwards obeys the same WLC law as the backbone.
    """
    if config.cohesion_max_total_neighbors == 0:
        return 0
    candidates = sorted(
        NeighborIndex(state.nodes_um, config.cohesion_distance_um).query_pairs(
            config.cohesion_distance_um
        )
    )
    original = {
        (min(int(a), int(b)), max(int(a), int(b))) for a, b in state.backbone_pairs
    }
    degree = cohesion_degrees(state)
    if config.cohesion_degree_includes_backbone:
        degree = degree + np.asarray(
            [len(neighbors) for neighbors in state.original_adjacency], dtype=int
        )
    new_this_search = np.zeros(len(state.nodes_um), dtype=int)
    created = 0
    for a, b in candidates:
        key = (min(a, b), max(a, b))
        if key in original or key in state.cohesion_pair_set:
            continue
        if config.cohesion_exclude_same_fiber and (
            state.node_fiber_ids[a] & state.node_fiber_ids[b]
        ):
            continue
        if degree[a] >= config.cohesion_max_total_neighbors or degree[b] >= config.cohesion_max_total_neighbors:
            continue
        if (
            new_this_search[a] >= config.cohesion_max_new_per_node_per_search
            or new_this_search[b] >= config.cohesion_max_new_per_node_per_search
        ):
            continue
        length = float(np.linalg.norm(state.nodes_um[b] - state.nodes_um[a]))
        if length <= 1e-12:
            continue
        state.cohesion_pairs.append(key)
        state.cohesion_L0_um.append(length)  # force is exactly zero at formation
        state.cohesion_birth_time_s.append(float(state.time_s))
        state.cohesion_pair_set.add(key)
        degree[a] += 1
        degree[b] += 1
        new_this_search[a] += 1
        new_this_search[b] += 1
        created += 1
    return created
