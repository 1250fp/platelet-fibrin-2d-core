"""Versioned state persistence for resumable Program19.3 mechanics."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .config import Program19Config
from .state import SimulationState, state_arrays


CHECKPOINT_SCHEMA_VERSION = 1


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def config_sha256(config: Program19Config) -> str:
    payload = json.dumps(
        config.to_dict(), sort_keys=True, separators=(",", ":"), default=_json_default
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: str | Path, payload: dict) -> None:
    """Write JSON through an adjacent temporary file before replacing the target."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target.stem}_", suffix=".json.tmp", dir=target.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_savez(path: str | Path, **arrays) -> None:
    """Atomically replace a compressed NPZ file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target.stem}_", suffix=".npz", dir=target.parent
    )
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def save_checkpoint(
    path: str | Path,
    state: SimulationState,
    rng: np.random.Generator,
    config: Program19Config,
    *,
    source: str,
) -> None:
    payload = state_arrays(state)
    payload.update(
        checkpoint_schema_version=np.asarray(CHECKPOINT_SCHEMA_VERSION),
        rng_bit_generator=np.asarray(type(rng.bit_generator).__name__),
        rng_state_json=np.asarray(
            json.dumps(rng.bit_generator.state, sort_keys=True, default=_json_default)
        ),
        config_sha256=np.asarray(config_sha256(config)),
        checkpoint_source=np.asarray(source),
    )
    noise_substep = getattr(state, "cuda_noise_substep", None)
    if noise_substep is not None:
        payload["cuda_noise_substep"] = np.asarray(np.uint64(noise_substep))
    atomic_savez(path, **payload)


def _adjacency_from_pairs(node_count: int, pairs: np.ndarray) -> tuple[list[list[int]], list[list[int]]]:
    incident: list[list[int]] = [[] for _ in range(node_count)]
    adjacency: list[list[int]] = [[] for _ in range(node_count)]
    for edge_id, (a_raw, b_raw) in enumerate(np.asarray(pairs, dtype=int).reshape((-1, 2))):
        a, b = int(a_raw), int(b_raw)
        if not (0 <= a < node_count and 0 <= b < node_count):
            raise ValueError(f"Backbone edge {(a, b)} is outside the node array")
        incident[a].append(edge_id)
        incident[b].append(edge_id)
        adjacency[a].append(b)
        adjacency[b].append(a)
    return incident, adjacency


def load_state(path: str | Path) -> SimulationState:
    """Reconstruct all derived state containers from a saved NPZ state."""
    with np.load(Path(path), allow_pickle=False) as saved:
        nodes = np.asarray(saved["nodes_um"], dtype=float).copy()
        platelets = np.asarray(saved["platelets_um"], dtype=float).copy()
        backbone = np.asarray(saved["backbone_pairs"], dtype=int).reshape((-1, 2)).copy()
        incident, adjacency = _adjacency_from_pairs(len(nodes), backbone)

        offsets = np.asarray(saved["node_fiber_id_offsets"], dtype=int)
        values = np.asarray(saved["node_fiber_ids_flat"], dtype=int)
        if len(offsets) != len(nodes) + 1:
            raise ValueError("Invalid node-fiber offset array")
        node_fibers = [
            frozenset(int(item) for item in values[offsets[i] : offsets[i + 1]])
            for i in range(len(nodes))
        ]

        raw_filopodia = np.asarray(saved["filopodia_node_ids"], dtype=int)
        if raw_filopodia.ndim != 2 or raw_filopodia.shape[0] != len(platelets):
            raise ValueError("Invalid filopodia array")
        filopodia = [
            [None if int(node) < 0 else int(node) for node in row]
            for row in raw_filopodia
        ]
        occupied: dict[int, tuple[int, int]] = {}
        for platelet_id, arms in enumerate(filopodia):
            for arm_id, node_id in enumerate(arms):
                if node_id is not None:
                    occupied[node_id] = (platelet_id, arm_id)

        cohesion_pairs_array = np.asarray(saved["cohesion_pairs"], dtype=int).reshape((-1, 2))
        cohesion_pairs = [tuple(map(int, pair)) for pair in cohesion_pairs_array]
        return SimulationState(
            nodes_um=nodes,
            platelets_um=platelets,
            backbone_pairs=backbone,
            backbone_L0_um=np.asarray(saved["backbone_reference_lengths_um"], dtype=float).copy(),
            bending_triples=np.asarray(saved["bending_triples"], dtype=int).reshape((-1, 3)).copy(),
            bending_phi0_rad=np.asarray(saved["bending_rest_angles_rad"], dtype=float).copy(),
            bending_dual_L0_um=np.asarray(
                saved["bending_dual_reference_lengths_um"], dtype=float
            ).copy(),
            node_fiber_ids=node_fibers,
            incident_backbone=incident,
            original_adjacency=adjacency,
            fixed_nodes=np.asarray(saved["fixed_nodes"], dtype=bool).copy(),
            cohesion_pairs=cohesion_pairs,
            cohesion_L0_um=list(
                map(float, np.asarray(saved["cohesion_reference_lengths_um"], dtype=float))
            ),
            cohesion_birth_time_s=list(
                map(float, np.asarray(saved["cohesion_birth_times_s"], dtype=float))
            ),
            cohesion_pair_set=set(cohesion_pairs),
            filopodia=filopodia,
            node_occupied_by=occupied,
            time_s=float(np.asarray(saved["time_s"]).item()),
            step=int(np.asarray(saved["step"]).item()),
        )


def load_rng(
    checkpoint_path: str | Path | None,
    *,
    fallback_seed: np.random.SeedSequence,
) -> tuple[np.random.Generator, str]:
    """Load an exact RNG continuation when available, otherwise create a recorded branch."""
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        with np.load(Path(checkpoint_path), allow_pickle=False) as saved:
            if "rng_state_json" in saved.files:
                generator = np.random.default_rng()
                generator.bit_generator.state = json.loads(str(saved["rng_state_json"].item()))
                return generator, "exact_saved_rng_state"
    return np.random.default_rng(fallback_seed), "legacy_reproducible_restart_stream"
