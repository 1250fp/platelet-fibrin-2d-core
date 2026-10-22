"""Reproducible multi-layer mechanical pipeline for the Program19.3 2-D core.

Pore analysis, tracer transport and the CodeOcean XML projection were removed
(they remain in platelet-fibrin-transport-lab-modernized).  Figures and raster
masks are optional; both need Matplotlib, which is imported lazily.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import threading
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from collections.abc import Callable

import numpy as np

from .config import Program19Config
from .geometry import build_initial_state
from .integrator import integrate_interval
from .layers import rasterize_geometry, rasterize_solid
from .parallel import (
    effective_worker_count,
    logical_cpu_count,
    numeric_thread_limit,
    run_process_jobs,
    throttled_progress_relay,
)
from .profiles import profile_audit
from .persistence import save_checkpoint
from .state import SimulationState, StepDiagnostics, state_arrays


ProgressCallback = Callable[[dict], None]


def _default_output_root() -> Path:
    return Path(__file__).resolve().parents[1] / "results"


def _new_run_directory(config: Program19Config) -> Path:
    root = Path(config.output_root).expanduser() if config.output_root else _default_output_root()
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = (
        f"program19_3_{config.mechanical_layers}mechanical_layers_"
        f"{config.equation_profile}_{config.parameter_profile}_"
        f"{config.numerical_profile}_seed{config.seed}_{timestamp}"
    )
    candidate = root / base
    suffix = 1
    while candidate.exists():
        candidate = root / f"{base}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dependency_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in ("numpy", "scipy", "matplotlib"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _write_manifest(
    run_dir: Path, config: Program19Config, figures: bool = True, masks: bool = True
) -> None:
    package_root = Path(__file__).resolve().parent
    source_paths = list(package_root.glob("*.py")) + [
        package_root.parent / "program19.py",
        package_root.parent / "program19_2.py",
        package_root.parent / "program19_3.py",
    ]
    source_paths = [path for path in source_paths if path.exists()]
    sources = {
        str(path.relative_to(package_root.parent)): _sha256(path)
        for path in sorted(source_paths)
    }
    manifest = {
        "program": "Program19.3",
        "version": "0.5.0",
        "created_local": datetime.now().isoformat(timespec="seconds"),
        "profiles": {
            "equation": config.equation_profile,
            "parameter": config.parameter_profile,
            "numerical": config.numerical_profile,
            "geometry": config.geometry_profile,
        },
        "profile_audit": profile_audit(config),
        # Static: the 2-D core only builds generated_2d geometry.
        "codeocean_source": {"used": False},
        "outputs": {"figures": bool(figures), "masks": bool(masks)},
        "configuration": config.to_dict(),
        "units": {"length": "um", "force": "nN", "time": "s"},
        "axis_order_for_stacks": "z,y,x",
        "layer_seed_rule": "layer_seed = base_seed + zero_based_layer_index*layer_seed_stride",
        "stage_semantics": {
            "mechanics": "N explicit 2-D mechanical realizations, one per z layer",
            "stack": "final rasterized mechanical mask from each layer",
            "autoregression": "not used",
            "interlayer_force_coupling": "none; layers evolve independently",
            "intermediate_states": (
                "positions/topology and optional raster masks saved at output_interval_s "
                "inside each layer folder"
            ),
            "parallelism": (
                "independent mechanical layers may run in separate processes; this changes "
                "wall time, not the per-layer equations, seed rule, or interlayer coupling"
            ),
        },
        "execution": {
            "mechanical_workers_requested": config.mechanical_workers,
            "mechanical_workers_effective": effective_worker_count(
                config.mechanical_workers, config.mechanical_layers
            ),
            "logical_cpu_count_detected": logical_cpu_count(),
            "worker_type": "separate Python processes",
            "numeric_library_threads_per_worker": 1,
        },
        "equations": {
            "strain": "gamma=(L-L0)/L0; u=gamma/C",
            "wlc": "F=N*kBT/P*[1/(4*(1-gamma/C)^2)-1/4+gamma/C]",
            "paper_stiffness": "dF/dgamma=N*kBT/P*[C^2/(2*(C-gamma)^3)+1/C]",
            "bending": "E=0.5*(EI/h0)*(phi-phi0)^2; F=-grad(E)",
            "analyzed_region": "per-layer padded convex support of the final solid phase",
        },
        "interpretation_warning": (
            "The stack is mechanically generated but is not a coupled 3-D mechanical model. "
            "No WLC, bending, cohesion, contact, or filopodial force acts between layers."
        ),
        "dependencies": _dependency_versions(),
        "platform": {"python": sys.version, "platform": platform.platform()},
        "source_sha256": sources,
    }
    with (run_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    config.save_json(run_dir / "config.json")
    with (run_dir / "profile_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(profile_audit(config), handle, indent=2, ensure_ascii=False)


def _filopodia_snapshot(state: SimulationState) -> np.ndarray:
    maximum = max((len(arms) for arms in state.filopodia), default=0)
    result = np.full((len(state.filopodia), maximum), -1, dtype=int)
    for platelet_id, arms in enumerate(state.filopodia):
        for arm_id, node_id in enumerate(arms):
            if node_id is not None:
                result[platelet_id, arm_id] = int(node_id)
    return result


def _flatten_cohesion_frames(
    frames: list[np.ndarray], reference_frames: list[np.ndarray], birth_frames: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    offsets = [0]
    for frame in frames:
        offsets.append(offsets[-1] + len(frame))
    pairs = np.concatenate(frames, axis=0) if offsets[-1] else np.empty((0, 2), dtype=int)
    references = np.concatenate(reference_frames) if offsets[-1] else np.empty(0)
    births = np.concatenate(birth_frames) if offsets[-1] else np.empty(0)
    return np.asarray(offsets, dtype=int), pairs, references, births


def _progress(
    callback: ProgressCallback | None, stage: str, fraction: float, **values
) -> None:
    if callback:
        callback({"type": "progress", "stage": stage, "fraction": float(fraction), **values})


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_trajectory(
    path: Path,
    state: SimulationState,
    config: Program19Config,
    frame_times: list[float],
    node_frames: list[np.ndarray],
    platelet_frames: list[np.ndarray],
    cohesion_frames: list[np.ndarray],
    cohesion_reference_frames: list[np.ndarray],
    cohesion_birth_frames: list[np.ndarray],
    filopodia_frames: list[np.ndarray],
) -> None:
    trajectory = {
        "times_s": np.asarray(frame_times),
        "nodes_um": np.asarray(node_frames),
        "platelets_um": np.asarray(platelet_frames),
        "backbone_pairs": state.backbone_pairs,
        "fixed_nodes": np.asarray(state.fixed_nodes, dtype=bool),
        "axis_order_for_positions": np.asarray("time,object,xy"),
    }
    if config.trajectory_detail == "topology":
        offsets, pairs, references, births = _flatten_cohesion_frames(
            cohesion_frames, cohesion_reference_frames, cohesion_birth_frames
        )
        maximum_arms = max((frame.shape[1] for frame in filopodia_frames), default=0)
        padded_filopodia = np.full(
            (len(filopodia_frames), len(state.platelets_um), maximum_arms), -1, dtype=int
        )
        for frame_id, frame in enumerate(filopodia_frames):
            padded_filopodia[frame_id, :, : frame.shape[1]] = frame
        trajectory.update(
            cohesion_frame_offsets=offsets,
            cohesion_pairs_flat=pairs,
            cohesion_reference_lengths_um_flat=references,
            cohesion_birth_times_s_flat=births,
            filopodia_node_ids=padded_filopodia,
        )
    np.savez_compressed(path, **trajectory)


def _save_intermediate_masks(
    path: Path,
    frame_times: list[float],
    node_frames: list[np.ndarray],
    platelet_frames: list[np.ndarray],
    backbone_pairs: np.ndarray,
    config: Program19Config,
) -> None:
    solid_masks: list[np.ndarray] = []
    region_masks: list[np.ndarray] = []
    extent = None
    dx = None
    for nodes, platelets in zip(node_frames, platelet_frames):
        solid, region, extent, dx = rasterize_geometry(nodes, platelets, backbone_pairs, config)
        solid_masks.append(solid)
        region_masks.append(region)
    np.savez_compressed(
        path,
        times_s=np.asarray(frame_times),
        solid_masks=np.asarray(solid_masks, dtype=bool),
        analyzed_region_masks=np.asarray(region_masks, dtype=bool),
        extent_um=np.asarray(extent),
        dx_um=np.asarray(dx),
        axis_order=np.asarray("time,y,x"),
        source=np.asarray("explicit mechanical states; no autoregression"),
    )


def _run_mechanical_layer(
    config: Program19Config,
    layer_index: int,
    layer_count: int,
    layer_dir: Path,
    progress_callback: ProgressCallback | None,
    pause_event: threading.Event,
    cancel_event: threading.Event,
    figures: bool = True,
    masks: bool = True,
) -> tuple[SimulationState, str, np.ndarray | None, np.ndarray | None, np.ndarray | None,
           float | None, dict]:
    """Run and persist one independent explicit mechanical layer."""
    layer_dir.mkdir(parents=True, exist_ok=False)
    config.save_json(layer_dir / "config.json")
    state = build_initial_state(config)
    np.savez_compressed(layer_dir / "initial_state.npz", **state_arrays(state))
    rng_mechanics = np.random.default_rng(np.random.SeedSequence([config.seed, 1001]))
    frame_times = [0.0]
    node_frames = [state.nodes_um.copy()]
    platelet_frames = [state.platelets_um.copy()]
    cohesion_frames = [np.asarray(state.cohesion_pairs, dtype=int).reshape((-1, 2))]
    cohesion_reference_frames = [np.asarray(state.cohesion_L0_um, dtype=float)]
    cohesion_birth_frames = [np.asarray(state.cohesion_birth_time_s, dtype=float)]
    filopodia_frames = [_filopodia_snapshot(state)]
    rows: list[dict] = []
    layer_wall_start = time.perf_counter()
    next_output = config.output_interval_s
    last_diagnostics = StepDiagnostics()
    status = "running"
    error: Exception | None = None
    try:
        while state.time_s < config.duration_s - 1e-15:
            if cancel_event.is_set():
                status = "cancelled"
                break
            while not pause_event.is_set():
                if cancel_event.wait(0.05):
                    status = "cancelled"
                    break
            if status == "cancelled":
                break
            interval = min(config.dt_s, config.duration_s - state.time_s)
            last_diagnostics = integrate_interval(state, config, rng_mechanics, interval)
            should_store = state.time_s >= next_output - 1e-12 or math.isclose(
                state.time_s, config.duration_s
            )
            if should_store:
                rows.append(
                    {
                        "step": state.step,
                        "simulated_time_s": state.time_s,
                        "wall_elapsed_s": time.perf_counter() - layer_wall_start,
                        **asdict(last_diagnostics),
                    }
                )
                frame_times.append(state.time_s)
                node_frames.append(state.nodes_um.copy())
                platelet_frames.append(state.platelets_um.copy())
                cohesion_frames.append(
                    np.asarray(state.cohesion_pairs, dtype=int).reshape((-1, 2)).copy()
                )
                cohesion_reference_frames.append(
                    np.asarray(state.cohesion_L0_um, dtype=float).copy()
                )
                cohesion_birth_frames.append(
                    np.asarray(state.cohesion_birth_time_s, dtype=float).copy()
                )
                filopodia_frames.append(_filopodia_snapshot(state))
                save_checkpoint(
                    layer_dir / "checkpoint.npz",
                    state,
                    rng_mechanics,
                    config,
                    source="Program19.3 periodic mechanical checkpoint",
                )
                while next_output <= state.time_s + 1e-12:
                    next_output += config.output_interval_s
            layer_fraction = state.time_s / config.duration_s
            _progress(
                progress_callback,
                "mechanics",
                (layer_index + layer_fraction) / layer_count,
                layer_index=layer_index + 1,
                layer_count=layer_count,
                layer_fraction=layer_fraction,
                simulated_time_s=state.time_s,
                duration_s=config.duration_s,
                cohesion_links=len(state.cohesion_pairs),
                internal_substeps=last_diagnostics.internal_substeps,
                layer_directory=str(layer_dir),
            )
        if status != "cancelled":
            status = "completed"
    except Exception as exc:
        status = "failed"
        error = exc

    np.savez_compressed(layer_dir / "final_state.npz", **state_arrays(state))
    save_checkpoint(
        layer_dir / "checkpoint.npz",
        state,
        rng_mechanics,
        config,
        source="Program19.3 final mechanical checkpoint",
    )
    if config.save_trajectory:
        _save_trajectory(
            layer_dir / "trajectory.npz", state, config, frame_times, node_frames,
            platelet_frames, cohesion_frames, cohesion_reference_frames,
            cohesion_birth_frames, filopodia_frames,
        )
    if masks and config.save_intermediate_masks:
        _save_intermediate_masks(
            layer_dir / "intermediate_masks.npz", frame_times, node_frames,
            platelet_frames, state.backbone_pairs, config,
        )
    _write_csv(layer_dir / "diagnostics.csv", rows)
    if figures:
        from .reporting import save_mechanics_figure

        save_mechanics_figure(layer_dir / "final_mechanics.png", state, config, status)
    solid = region = extent = dx = None
    if masks:
        solid, region, extent, dx = rasterize_solid(state, config)
        np.savez_compressed(
            layer_dir / "final_masks.npz",
            solid_mask=solid,
            analyzed_region_mask=region,
            extent_um=extent,
            dx_um=np.asarray(dx),
            simulated_time_s=np.asarray(state.time_s),
            source=np.asarray("explicit mechanical final state; no autoregression"),
        )
        if figures:
            from .reporting import save_region_figure

            save_region_figure(layer_dir / "analyzed_region_2d.png", solid, region, extent)
    layer_summary = {
        "layer_index_one_based": layer_index + 1,
        "z_um": layer_index * config.layer_spacing_um,
        "seed": config.seed,
        "status": status,
        "simulated_time_s": state.time_s,
        "completed_outer_steps": state.step,
        "saved_intermediate_states": len(frame_times),
        "node_count": len(state.nodes_um),
        "platelet_count": len(state.platelets_um),
        "cohesion_link_count": len(state.cohesion_pairs),
        "solid_pixels": int(np.count_nonzero(solid & region)) if masks else None,
        "analyzed_region_pixels": int(np.count_nonzero(region)) if masks else None,
        "geometric_porosity_2d": (
            float(np.count_nonzero(region & ~solid) / max(np.count_nonzero(region), 1))
            if masks else None
        ),
        "wall_elapsed_s": time.perf_counter() - layer_wall_start,
        "worker_process_id": os.getpid(),
        "mechanical_workers_requested": config.mechanical_workers,
        "mechanics_backend": config.mechanics_backend,
        "error": str(error) if error else None,
    }
    with (layer_dir / "layer_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(layer_summary, handle, indent=2)
    if error:
        raise RuntimeError(f"Layer {layer_index + 1} failed: {error}") from error
    return state, status, solid, region, extent, dx, layer_summary


def _parallel_mechanical_layer_worker(
    layer_index: int,
    config_values: dict,
    layer_count: int,
    layer_dir_text: str,
    figures: bool,
    masks: bool,
    pause_event,
    cancel_event,
    progress_queue,
):
    """Spawn-safe entry point for one independently simulated layer."""
    config = Program19Config.from_dict(config_values)
    relay = throttled_progress_relay(progress_queue)
    with numeric_thread_limit():
        return _run_mechanical_layer(
            config,
            layer_index,
            layer_count,
            Path(layer_dir_text),
            relay,
            pause_event,
            cancel_event,
            figures,
            masks,
        )


def _require_matplotlib(figures: bool, masks: bool) -> None:
    if not (figures or masks):
        return
    import importlib.util

    if importlib.util.find_spec("matplotlib") is None:
        raise ImportError(
            "Matplotlib is needed for figures and raster masks; install it "
            "or run with --no-figures --no-masks"
        )


def run_pipeline(
    config: Program19Config,
    progress_callback: ProgressCallback | None = None,
    pause_event: threading.Event | None = None,
    cancel_event: threading.Event | None = None,
    *,
    figures: bool = True,
    masks: bool = True,
) -> tuple[SimulationState, Path, str]:
    """Run every mechanical layer and write its outputs.

    ``figures`` writes the PNGs; ``masks`` writes final_masks.npz,
    intermediate_masks.npz (if config.save_intermediate_masks) and the
    mechanical_spatial_stack.npz.  Neither affects the mechanics.
    """
    config.validate()
    if config.run_transport:
        raise ValueError(
            "run_transport (first-passage transport) is not part of the 2-D core; "
            "use the platelet-fibrin-transport-lab-modernized folder"
        )
    if config.geometry_profile != "generated_2d":
        raise ValueError(
            "only geometry_profile='generated_2d' is supported by the 2-D core"
        )
    _require_matplotlib(figures, masks)
    pause_event = pause_event or threading.Event()
    pause_event.set()
    cancel_event = cancel_event or threading.Event()
    run_dir = _new_run_directory(config)
    _write_manifest(run_dir, config, figures, masks)
    layers_dir = run_dir / "mechanical_layers"
    layers_dir.mkdir()
    wall_start = time.perf_counter()
    status = "running"
    error: Exception | None = None
    final_states: list[SimulationState] = []
    solid_layers: list[np.ndarray] = []
    region_layers: list[np.ndarray] = []
    layer_rows: list[dict] = []
    extent = None
    dx = None
    try:
        workers = effective_worker_count(config.mechanical_workers, config.mechanical_layers)
        if workers == 1:
            for layer_index in range(config.mechanical_layers):
                if cancel_event.is_set():
                    status = "cancelled"
                    break
                layer_seed = config.seed + layer_index * config.layer_seed_stride
                layer_config = replace(config, seed=layer_seed)
                layer_dir = layers_dir / f"layer_{layer_index + 1:03d}_seed{layer_seed}"
                state, layer_status, solid, region, extent, dx, summary = _run_mechanical_layer(
                    layer_config, layer_index, config.mechanical_layers, layer_dir,
                    progress_callback, pause_event, cancel_event, figures, masks,
                )
                final_states.append(state)
                solid_layers.append(solid)
                region_layers.append(region)
                layer_rows.append({**summary, "folder": str(layer_dir.relative_to(run_dir))})
                _write_csv(run_dir / "layer_index.csv", layer_rows)
                if layer_status == "cancelled":
                    status = "cancelled"
                    break
        else:
            completed_results: dict[int, tuple] = {}

            def accept_result(layer_index: int, result: object) -> None:
                nonlocal extent, dx
                completed_results[layer_index] = result  # type: ignore[assignment]
                ordered = [completed_results[index] for index in sorted(completed_results)]
                final_states[:] = [item[0] for item in ordered]
                solid_layers[:] = [item[2] for item in ordered]
                region_layers[:] = [item[3] for item in ordered]
                extent = ordered[-1][4]
                dx = ordered[-1][5]
                layer_rows[:] = []
                for item in ordered:
                    summary = item[6]
                    layer_dir = layers_dir / (
                        f"layer_{int(summary['layer_index_one_based']):03d}_"
                        f"seed{int(summary['seed'])}"
                    )
                    layer_rows.append(
                        {**summary, "folder": str(layer_dir.relative_to(run_dir))}
                    )
                _write_csv(run_dir / "layer_index.csv", layer_rows)

            jobs = []
            for layer_index in range(config.mechanical_layers):
                layer_seed = config.seed + layer_index * config.layer_seed_stride
                layer_config = replace(config, seed=layer_seed)
                layer_dir = layers_dir / f"layer_{layer_index + 1:03d}_seed{layer_seed}"
                jobs.append(
                    (
                        layer_index,
                        layer_config.to_dict(),
                        config.mechanical_layers,
                        str(layer_dir),
                        figures,
                        masks,
                    )
                )
            run_process_jobs(
                _parallel_mechanical_layer_worker,
                jobs,
                requested_workers=config.mechanical_workers,
                layer_count=config.mechanical_layers,
                stage="mechanics",
                progress_callback=progress_callback,
                result_callback=accept_result,
                pause_event=pause_event,
                cancel_event=cancel_event,
            )
            if cancel_event.is_set() or any(
                row.get("status") == "cancelled" for row in layer_rows
            ) or len(layer_rows) < config.mechanical_layers:
                status = "cancelled"
        if status == "running":
            status = "completed"

        if not final_states:
            raise RuntimeError("No mechanical layer reached a saveable state")
        if masks:
            _progress(
                progress_callback, "stack_assembly", 0.90,
                completed_layers=len(solid_layers), requested_layers=config.mechanical_layers,
                run_directory=str(run_dir),
            )
            stack = np.asarray(solid_layers, dtype=bool)
            region_stack = np.asarray(region_layers, dtype=bool)
            layer_seeds = np.asarray([row["seed"] for row in layer_rows], dtype=int)
            layer_times = np.asarray(
                [row["simulated_time_s"] for row in layer_rows], dtype=float
            )
            np.savez_compressed(
                run_dir / "mechanical_spatial_stack.npz",
                solid_layers=stack,
                analyzed_region_masks=region_stack,
                extent_um=np.asarray(extent),
                dx_um=np.asarray(dx),
                dz_um=np.asarray(config.layer_spacing_um),
                layer_z_um=np.arange(len(stack), dtype=float) * config.layer_spacing_um,
                layer_seeds=layer_seeds,
                simulated_time_s_by_layer=layer_times,
                source=np.asarray(
                    "independent explicit mechanical layers; no autoregression; "
                    "no interlayer forces"
                ),
                axis_order=np.asarray("z,y,x"),
            )
        run_metrics = {
            "mechanical_layers_requested": config.mechanical_layers,
            "mechanical_layers_completed": len(final_states),
            "mechanical_workers_requested": config.mechanical_workers,
            "mechanical_workers_effective": effective_worker_count(
                config.mechanical_workers, config.mechanical_layers
            ),
            "mechanics_backend": config.mechanics_backend,
            "autoregression_used": False,
            "interlayer_force_coupling": False,
            "filopodia_f0_nN": config.filopodia_f0_nN,
            "outputs": {"figures": bool(figures), "masks": bool(masks)},
        }
        with (run_dir / "run_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(run_metrics, handle, indent=2)
        _progress(progress_callback, "complete", 1.0, run_directory=str(run_dir))
    except Exception as exc:
        status = "failed"
        error = exc
    finally:
        with (run_dir / "status.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "status": status,
                    "requested_mechanical_layers": config.mechanical_layers,
                    "completed_or_partial_layers_saved": len(final_states),
                    "mechanical_workers_requested": config.mechanical_workers,
                    "mechanical_workers_effective": effective_worker_count(
                        config.mechanical_workers, config.mechanical_layers
                    ),
                    "wall_elapsed_s": time.perf_counter() - wall_start,
                    "error": str(error) if error else None,
                },
                handle,
                indent=2,
            )
    if error:
        raise RuntimeError(f"{error} Partial results were saved in {run_dir}") from error
    return final_states[-1], run_dir, status
