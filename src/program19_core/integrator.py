"""Force-recomputed adaptive overdamped integrator with no clipping."""

from __future__ import annotations

import math
import numpy as np

from .bending import bending_force_field
from .cohesion import add_new_cohesion_links, cohesion_degrees
from .config import Program19Config
from .interactions import all_platelet_interactions
from .state import (
    ForceDiagnostics,
    SimulationState,
    StepDiagnostics,
    kbt_nN_um,
    max_vector_norm,
    mobility_and_diffusion,
)
from .wlc import backbone_and_cohesion_forces


def calculate_forces(
    state: SimulationState,
    config: Program19Config,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, ForceDiagnostics]:
    """Total deterministic force on every node and platelet (nN).

    Node force = WLC stretch (backbone + cohesion) + bending + platelet contact
    and filopodia; platelet force = reactions + platelet--platelet exclusion.
    """
    stretch, wlc_rms, max_strain = backbone_and_cohesion_forces(state, config)
    bending, bending_rms = bending_force_field(state, config)
    contact_nodes, contact_platelets, contact_rms, attached, overlaps, minimum = (
        all_platelet_interactions(state, config, rng)
    )
    node_force = stretch + bending + contact_nodes
    platelet_force = contact_platelets
    total = np.sum(node_force, axis=0) + np.sum(platelet_force, axis=0)
    return node_force, platelet_force, ForceDiagnostics(
        wlc_rms_nN=wlc_rms,
        bending_rms_nN=bending_rms,
        contact_rms_nN=contact_rms,
        filopodia_rms_nN=0.0,  # included in contact_rms; retained for stable schema
        max_backbone_strain=max_strain,
        max_total_node_force_nN=max_vector_norm(node_force),
        max_total_platelet_force_nN=max_vector_norm(platelet_force),
        net_internal_force_nN=float(np.linalg.norm(total)),
        attached_filopodia=attached,
        platelet_overlap_count=overlaps,
        minimum_platelet_center_distance_um=minimum,
    )


def integrate_interval(
    state: SimulationState,
    config: Program19Config,
    rng: np.random.Generator,
    interval_s: float,
) -> StepDiagnostics:
    """Integrate exactly one requested outer interval.

    Overdamped (Brownian) dynamics, explicit Euler--Maruyama per substep:

        dx = (F / zeta) * dt + sqrt(2 * D * dt) * xi,   D = k_B*T / zeta

    zeta is the drag (nN*s/um) from ``mobility_and_diffusion`` and xi a standard
    normal draw per coordinate (or the archived shared uniform draw). With
    ``adaptive_substeps`` the substep dt is shortened so the largest
    deterministic displacement is at most ``adaptive_target_displacement_um``;
    forces are recomputed every substep. Displacements are never clipped: a
    move above ``abort_displacement_um`` stops the run. The common translation
    is removed and fixed nodes are held. Cohesion links are searched every
    ``cohesion_search_stride`` outer steps.
    """
    if config.mechanics_backend == "cuda":
        from .cuda_mechanics import integrate_interval_cuda

        return integrate_interval_cuda(state, config, rng, interval_s)
    new_links = 0
    if state.step % config.cohesion_search_stride == 0:
        new_links = add_new_cohesion_links(state, config)
    node_drag, platelet_drag, node_D, platelet_D = mobility_and_diffusion(config)
    remaining = float(interval_s)
    substeps = 0
    maximum_node_move = 0.0
    maximum_platelet_move = 0.0
    maximum_node_force = 0.0
    current = ForceDiagnostics()
    while remaining > max(1e-15, interval_s * 1e-12):
        if substeps >= config.max_internal_substeps:
            raise RuntimeError(
                f"Adaptive integrator exceeded {config.max_internal_substeps} substeps "
                f"during an outer interval of {interval_s:g} s. No displacement was clipped."
            )
        node_force, platelet_force, current = calculate_forces(state, config, rng)
        maximum_node_force = max(maximum_node_force, current.max_total_node_force_nN)
        node_speed = current.max_total_node_force_nN / node_drag
        platelet_speed = current.max_total_platelet_force_nN / platelet_drag
        maximum_speed = max(node_speed, platelet_speed)
        internal_dt = remaining
        if (
            config.adaptive_substeps
            and maximum_speed * internal_dt > config.adaptive_target_displacement_um
        ):
            internal_dt = config.adaptive_target_displacement_um / maximum_speed
        if not math.isfinite(internal_dt) or internal_dt <= 0.0:
            raise FloatingPointError("Adaptive integration produced an invalid substep")
        node_move = internal_dt * node_force / node_drag
        platelet_move = internal_dt * platelet_force / platelet_drag
        if config.brownian_motion:
            if config.thermal_noise_model == "codeocean_uniform_shared":
                # Literal projected counterpart of psrunifgen(-1,1): one
                # scalar is reused in both coordinates of each object.
                node_move += rng.uniform(-1.0, 1.0, size=(len(state.nodes_um), 1)) * math.sqrt(
                    2.0 * node_D * internal_dt
                )
                platelet_move += rng.uniform(
                    -1.0, 1.0, size=(len(state.platelets_um), 1)
                ) * math.sqrt(2.0 * platelet_D * internal_dt)
            else:
                node_move += rng.normal(
                    0.0, math.sqrt(2.0 * node_D * internal_dt), size=state.nodes_um.shape
                )
                platelet_move += rng.normal(
                    0.0,
                    math.sqrt(2.0 * platelet_D * internal_dt),
                    size=state.platelets_um.shape,
                )
        if config.remove_global_translation:
            count = len(node_move) + len(platelet_move)
            if count:
                common = (np.sum(node_move, axis=0) + np.sum(platelet_move, axis=0)) / count
                node_move -= common
                platelet_move -= common
        if len(state.fixed_nodes):
            node_move[np.asarray(state.fixed_nodes, dtype=bool)] = 0.0
        node_max = max_vector_norm(node_move)
        platelet_max = max_vector_norm(platelet_move)
        requested = max(node_max, platelet_max)
        if not math.isfinite(requested):
            raise FloatingPointError("Integrator produced a non-finite displacement")
        if requested > config.abort_displacement_um:
            raise RuntimeError(
                f"Safety stop: a substep requested {requested:.6g} um, above "
                f"abort_displacement_um={config.abort_displacement_um:g}. No clip was applied."
            )
        state.nodes_um += node_move
        state.platelets_um += platelet_move
        maximum_node_move = max(maximum_node_move, node_max)
        maximum_platelet_move = max(maximum_platelet_move, platelet_max)
        remaining = max(0.0, remaining - internal_dt)
        substeps += 1
    state.time_s += float(interval_s)
    state.step += 1
    degree = cohesion_degrees(state)
    return StepDiagnostics(
        **current.__dict__,
        new_cohesion_links=new_links,
        total_cohesion_links=len(state.cohesion_pairs),
        maximum_cohesion_degree=int(np.max(degree)) if len(degree) else 0,
        max_node_displacement_um=maximum_node_move,
        max_platelet_displacement_um=maximum_platelet_move,
        internal_substeps=substeps,
        node_diffusion_um2_s=node_D,
        platelet_diffusion_um2_s=platelet_D,
    )
