"""Reciprocal platelet contacts and stiffness-dependent filopodia."""

from __future__ import annotations

import numpy as np

from .config import Program19Config
from .spatial import NeighborIndex
from .state import SimulationState, rms_vectors
from .wlc import local_node_stiffness_nN


def platelet_fibrin_contact_forces(
    state: SimulationState, config: Program19Config
) -> tuple[np.ndarray, np.ndarray]:
    """Active CUDA constant repulsion/adhesion shells with action--reaction.

    Around each platelet (radius R_p) a fibrin node closer than
    r_in = core_fraction * R_p + r_fibre is pushed outward with constant force
    ``platelet_fibrin_force_nN``; a node between r_in and R_p + r_fibre is pulled
    inward with the same magnitude (adhesion). The platelet receives the
    opposite force.
    """
    node_force = np.zeros_like(state.nodes_um)
    platelet_force = np.zeros_like(state.platelets_um)
    if not len(state.platelets_um) or not len(state.nodes_um):
        return node_force, platelet_force
    inner = config.platelet_core_fraction * config.platelet_radius_um + config.fiber_radius_um
    outer = config.platelet_radius_um + config.fiber_radius_um
    index = NeighborIndex(state.nodes_um, outer)
    magnitude0 = config.platelet_fibrin_force_nN
    for p_id, center in enumerate(state.platelets_um):
        for node_id in index.query_ball_point(center, outer):
            delta = state.nodes_um[node_id] - center
            distance = float(np.linalg.norm(delta))
            if distance <= 1e-12:
                raise RuntimeError("A fibrin node coincides with a platelet center")
            direction = delta / distance
            magnitude = magnitude0 if distance < inner else -magnitude0
            vector = magnitude * direction
            node_force[node_id] += vector
            platelet_force[p_id] -= vector
    return node_force, platelet_force


def _platelet_pair_scalar_nN(distance: float, config: Program19Config) -> float:
    """Radial 12-6 (Lennard-Jones-type) platelet--platelet force (nN; >0 repulsive).

    ``paper_consistent`` puts the zero crossing at the core diameter
    d = 2 * R_p * core_fraction (sigma = d / 2^(1/6)). Only pairs closer than
    2 * R_p are evaluated.
    """
    core_diameter = 2.0 * config.platelet_radius_um * config.platelet_core_fraction
    if config.equation_profile == "codeocean_literal":
        # Literal active CUDA branch: powf(2, 1/6) used integer division, hence 1.
        sigma = core_diameter
        q = sigma / distance
        return 12.0 * config.platelet_platelet_force_parameter**2 * (q**12 - q**6) / distance
    # Dimensionally labelled 12-6 derivative. Its zero is the intended core diameter.
    sigma = core_diameter / 2.0 ** (1.0 / 6.0)
    q = sigma / distance
    epsilon_nN_um = config.platelet_platelet_force_parameter
    return 24.0 * epsilon_nN_um * (2.0 * q**12 - q**6) / distance


def platelet_platelet_contact_forces(
    state: SimulationState, config: Program19Config
) -> tuple[np.ndarray, int, float]:
    """The active volume-exclusion branch omitted in the earlier 2-D port."""
    force = np.zeros_like(state.platelets_um)
    if len(state.platelets_um) < 2:
        return force, 0, float("inf")
    cutoff = 2.0 * config.platelet_radius_um
    pairs = sorted(NeighborIndex(state.platelets_um, cutoff).query_pairs(cutoff))
    overlap_count = 0
    minimum = float("inf")
    for a, b in pairs:
        delta = state.platelets_um[b] - state.platelets_um[a]
        distance = float(np.linalg.norm(delta))
        minimum = min(minimum, distance)
        if distance <= 1e-9:
            raise RuntimeError("Two platelet centers coincide; contact force is singular")
        if distance < cutoff:
            overlap_count += 1
        scalar = _platelet_pair_scalar_nN(distance, config)
        vector_on_a = -scalar * delta / distance
        force[a] += vector_on_a
        force[b] -= vector_on_a
    if not pairs:
        # A separate nearest-neighbour calculation keeps the diagnostic meaningful.
        delta = state.platelets_um[:, None, :] - state.platelets_um[None, :, :]
        distances = np.linalg.norm(delta, axis=2)
        distances[distances == 0.0] = np.inf
        minimum = float(np.min(distances))
    return force, overlap_count, minimum


def filopodial_force_nN(local_stiffness_nN: float, config: Program19Config) -> float:
    """Stiffness-dependent filopodial pulling force per arm (nN).

        F = max(0, 1/2 * (f0 + a * k / (b * k + c)))

    k = local fibre stiffness (nN), f0 = ``filopodia_f0_nN``; a, b, c are the
    fitted ``filopodia_a/b/c`` constants. Stiffer fibres are pulled harder,
    saturating at large k.
    """
    denominator = config.filopodia_b * local_stiffness_nN + config.filopodia_c
    response = 0.0 if abs(denominator) <= 1e-12 else (
        config.filopodia_a * local_stiffness_nN / denominator
    )
    # The active CUDA functor clamps a negative result to zero.
    return max(0.0, 0.5 * (config.filopodia_f0_nN + response))


def _cohesive_neighbors(state: SimulationState, node_id: int) -> list[int]:
    result = list(state.original_adjacency[node_id])
    for a, b in state.cohesion_pairs:
        if a == node_id:
            result.append(b)
        elif b == node_id:
            result.append(a)
    return result


def _available_for_arm(
    state: SimulationState,
    config: Program19Config,
    arms: list[int | None],
    candidate: int,
) -> bool:
    if candidate in arms:
        return False
    return not (
        config.filopodia_global_node_exclusivity
        and candidate in state.node_occupied_by
    )


def _claim_node(
    state: SimulationState,
    config: Program19Config,
    node_id: int,
    platelet_id: int,
    arm_id: int,
) -> None:
    if config.filopodia_global_node_exclusivity:
        state.node_occupied_by[int(node_id)] = (platelet_id, arm_id)


def filopodia_forces_and_update(
    state: SimulationState,
    config: Program19Config,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Update filopodial attachments, then return their forces and the attached count.

    Arms attach to free fibrin nodes in the annulus [R_p + r_fibre,
    ``platelet_force_radius_um``] and release when the node leaves it. Each
    attached arm pulls node and platelet towards each other with
    ``filopodial_force_nN`` (optionally divided among attached arms).
    """
    node_force = np.zeros_like(state.nodes_um)
    platelet_force = np.zeros_like(state.platelets_um)
    if not len(state.platelets_um) or not len(state.nodes_um):
        return node_force, platelet_force, 0
    inner = config.platelet_radius_um + config.fiber_radius_um
    outer = config.platelet_force_radius_um
    index = NeighborIndex(state.nodes_um, outer)

    rng = rng or np.random.default_rng(0)

    # Release arms that left the audited annular zone.  The CodeOcean-like
    # profile first attempts a graph-neighbor hand-over from the previous node.
    for p_id, arms in enumerate(state.filopodia):
        center = state.platelets_um[p_id]
        for arm_id, node_id in enumerate(arms):
            if node_id is None:
                continue
            distance = float(np.linalg.norm(state.nodes_um[node_id] - center))
            if not (inner <= distance <= outer):
                arms[arm_id] = None
                state.node_occupied_by.pop(int(node_id), None)
                if config.filopodia_handover_rule == "graph_neighbor":
                    neighbors = _cohesive_neighbors(state, int(node_id))
                    if config.filopodia_selection_rule == "codeocean_random":
                        rng.shuffle(neighbors)
                    else:
                        neighbors.sort(
                            key=lambda node: float(
                                np.linalg.norm(state.nodes_um[node] - center)
                            )
                        )
                    for candidate in neighbors:
                        candidate_distance = float(
                            np.linalg.norm(state.nodes_um[candidate] - center)
                        )
                        if (
                            inner <= candidate_distance <= outer
                            and _available_for_arm(state, config, arms, candidate)
                        ):
                            arms[arm_id] = candidate
                            _claim_node(state, config, candidate, p_id, arm_id)
                            break

    # Attach free arms to the nearest free node and recheck the same annulus.
    for p_id, arms in enumerate(state.filopodia):
        center = state.platelets_um[p_id]
        candidates = index.query_ball_point(center, outer)
        if config.filopodia_selection_rule == "codeocean_random":
            rng.shuffle(candidates)
        else:
            candidates.sort(key=lambda node: float(np.linalg.norm(state.nodes_um[node] - center)))
        for arm_id, node_id in enumerate(arms):
            if node_id is not None:
                continue
            for candidate in candidates:
                distance = float(np.linalg.norm(state.nodes_um[candidate] - center))
                if (
                    inner <= distance <= outer
                    and _available_for_arm(state, config, arms, candidate)
                ):
                    arms[arm_id] = candidate
                    _claim_node(state, config, candidate, p_id, arm_id)
                    break

    attached = 0
    for p_id, arms in enumerate(state.filopodia):
        center = state.platelets_um[p_id]
        attached_nodes = [node_id for node_id in arms if node_id is not None]
        divisor = (
            max(len(attached_nodes), 1)
            if config.filopodia_force_distribution == "divide_by_attached"
            else 1
        )
        for node_id in attached_nodes:
            if node_id is None:
                continue
            delta = center - state.nodes_um[node_id]
            distance = float(np.linalg.norm(delta))
            if distance <= 1e-12:
                continue
            magnitude = filopodial_force_nN(
                local_node_stiffness_nN(state, int(node_id), config), config
            ) / divisor
            vector = magnitude * delta / distance
            node_force[node_id] += vector
            platelet_force[p_id] -= vector
            attached += 1
    return node_force, platelet_force, attached


def all_platelet_interactions(
    state: SimulationState,
    config: Program19Config,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, float, int, int, float]:
    zone_nodes, zone_platelets = platelet_fibrin_contact_forces(state, config)
    pair_platelets, overlaps, minimum = platelet_platelet_contact_forces(state, config)
    arm_nodes, arm_platelets, attached = filopodia_forces_and_update(
        state, config, rng
    )
    node_force = zone_nodes + arm_nodes
    platelet_force = zone_platelets + pair_platelets + arm_platelets
    combined = np.vstack((node_force, platelet_force))
    return node_force, platelet_force, rms_vectors(combined), attached, overlaps, minimum
