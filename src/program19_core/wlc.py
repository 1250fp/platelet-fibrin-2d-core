"""Audited Michael/Marko--Siggia stretch law and local tangent."""

from __future__ import annotations

import numpy as np

from .config import Program19Config
from .state import J_PER_NN_UM, K_B_J_PER_K, SimulationState, kbt_nN_um, rms_vectors


def engineering_strain(distance_um, reference_um, config: Program19Config):
    """Engineering strain gamma = (L - L0) / L0 of a segment, capped above at gamma_max.

    L is the current length and L0 the rest length (both um; L0 floored at 1e-12 um).
    Negative values (compression) are kept; only the upper end is capped.
    """
    distance = np.asarray(distance_um, dtype=float)
    reference = np.maximum(np.asarray(reference_um, dtype=float), 1e-12)
    return np.minimum((distance - reference) / reference, config.gamma_max)


def wlc_force_nN(distance_um, reference_um, config: Program19Config):
    """F(gamma), where gamma=(L-L0)/L0 and u=gamma/C.

    Marko--Siggia worm-like-chain interpolation, scaled by the number of
    protofibrils in parallel:

        F = (N * k_B*T / P) * [ 1 / (4 * (1 - u)^2) - 1/4 + u ]     (nN)

    N = ``monomers_in_parallel_N``, P = ``persistence_length_um`` (um),
    C = ``contour_multiplier_C`` (contour length / rest length). F is zero at
    gamma = 0, positive (tension) when stretched and diverges as u -> 1; the
    denominator is floored at 1e-12 for safety.

    Compression is signed. Only the upper constitutive evaluation is capped;
    coordinates are never projected and displacements are never clipped.
    """
    gamma = engineering_strain(distance_um, reference_um, config)
    normalized = gamma / config.contour_multiplier_C
    denominator = np.maximum(1.0 - normalized, 1e-12)
    prefactor = (
        config.monomers_in_parallel_N
        * kbt_nN_um(config)
        / config.persistence_length_um
    )
    result = prefactor * (0.25 / denominator**2 - 0.25 + normalized)
    return float(result) if np.ndim(result) == 0 else result


def paper_stiffness_dF_dgamma_nN(strain, config: Program19Config):
    """Analytical tangent stiffness dF/dgamma of ``wlc_force_nN`` (nN per unit strain).

        dF/dgamma = (N * k_B*T / P) * [ C^2 / (2 * (C - gamma)^3) + 1/C ]

    Used as the "local fibre stiffness" that sets the filopodial pulling force.
    """
    gamma = np.minimum(np.asarray(strain, dtype=float), config.gamma_max)
    C = config.contour_multiplier_C
    denominator = np.maximum(C - gamma, 1e-12)
    prefactor = (
        config.monomers_in_parallel_N
        * kbt_nN_um(config)
        / config.persistence_length_um
    )
    result = prefactor * (C**2 / (2.0 * denominator**3) + 1.0 / C)
    return float(result) if np.ndim(result) == 0 else result


def archive_literal_local_stiffness_nN(strain, config: Program19Config):
    """Literal audited CUDA stiffness convention, not an analytical derivative.

    The archive averages absolute strain and uses a first-power denominator in
    its second term. This profile exists for reproduction and sensitivity only.
    """
    gamma = np.minimum(np.abs(np.asarray(strain, dtype=float)), config.gamma_max)
    C = config.contour_multiplier_C
    denominator = np.maximum(C - gamma, 1e-12)
    prefactor = (
        config.monomers_in_parallel_N
        * kbt_nN_um(config)
        / config.persistence_length_um
    )
    result = prefactor * (1.0 / C + C**2 / (2.0 * denominator))
    return float(result) if np.ndim(result) == 0 else result


def pair_force_field(
    nodes: np.ndarray,
    pairs: np.ndarray,
    reference_lengths: np.ndarray,
    config: Program19Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Central WLC forces for a list of node pairs (backbone or cohesion links).

    Each pair (a, b) carries F(gamma) along the unit vector from a to b. A
    positive (tensile) force pulls a towards b and b towards a with equal and
    opposite vectors, so the pair force sums to zero. Returns per-node forces
    (nN), per-pair strains and per-pair force magnitudes (nN).
    """
    force = np.zeros_like(nodes)
    if len(pairs) == 0:
        return force, np.empty(0), np.empty(0)
    delta = nodes[pairs[:, 1]] - nodes[pairs[:, 0]]
    distance = np.linalg.norm(delta, axis=1)
    safe = np.maximum(distance, 1e-12)
    magnitude = wlc_force_nN(distance, reference_lengths, config)
    vector = magnitude[:, None] * delta / safe[:, None]
    np.add.at(force, pairs[:, 0], vector)
    np.add.at(force, pairs[:, 1], -vector)
    strain = engineering_strain(distance, reference_lengths, config)
    return force, strain, magnitude


def backbone_and_cohesion_forces(
    state: SimulationState, config: Program19Config
) -> tuple[np.ndarray, float, float]:
    """Total stretch force from fibre backbone segments plus cohesion links.

    Returns per-node force (nN), its RMS and the largest absolute backbone strain.
    """
    backbone, strain, _ = pair_force_field(
        state.nodes_um, state.backbone_pairs, state.backbone_L0_um, config
    )
    total = backbone
    if state.cohesion_pairs:
        pairs = np.asarray(state.cohesion_pairs, dtype=int).reshape((-1, 2))
        cohesive, _, _ = pair_force_field(
            state.nodes_um, pairs, np.asarray(state.cohesion_L0_um), config
        )
        total = total + cohesive
    maximum = float(np.max(np.abs(strain))) if len(strain) else 0.0
    return total, rms_vectors(total), maximum


def local_node_stiffness_nN(
    state: SimulationState, node_index: int, config: Program19Config
) -> float:
    """Stiffness (nN) at one node from the mean strain of its incident backbone segments."""
    incident = state.incident_backbone[node_index]
    if not incident:
        return 0.0
    edge_ids = np.asarray(incident, dtype=int)
    pairs = state.backbone_pairs[edge_ids]
    distance = np.linalg.norm(
        state.nodes_um[pairs[:, 1]] - state.nodes_um[pairs[:, 0]], axis=1
    )
    strain = (distance - state.backbone_L0_um[edge_ids]) / np.maximum(
        state.backbone_L0_um[edge_ids], 1e-12
    )
    if config.equation_profile == "codeocean_literal":
        return float(archive_literal_local_stiffness_nN(np.mean(np.abs(strain)), config))
    return float(paper_stiffness_dF_dgamma_nN(np.mean(strain), config))
