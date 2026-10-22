"""Deterministic report figures produced directly from Program19 outputs."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from .config import Program19Config
from .state import SimulationState


def _pyplot():
    """Import Matplotlib only when a figure is written (figures are optional).

    The non-interactive Agg backend is selected before pyplot is imported, so
    spawned layer processes never create a GUI window.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_mechanics_figure(
    path: Path, state: SimulationState, config: Program19Config, status: str
) -> None:
    plt = _pyplot()
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle

    figure, axis = plt.subplots(figsize=(7.2, 7.2), dpi=170)
    if len(state.backbone_pairs):
        axis.add_collection(
            LineCollection(
                state.nodes_um[state.backbone_pairs],
                colors="#2b4c6f",
                linewidths=0.8,
                alpha=0.95,
            )
        )
    if state.cohesion_pairs:
        pairs = np.asarray(state.cohesion_pairs, dtype=int)
        axis.add_collection(
            LineCollection(
                state.nodes_um[pairs], colors="#d95f02", linewidths=1.0, alpha=0.9
            )
        )
    for center in state.platelets_um:
        axis.add_patch(
            Circle(
                center,
                config.platelet_radius_um,
                facecolor="#ceb07a",
                edgecolor="#5b4630",
                linewidth=0.7,
            )
        )
    half = 0.53 * config.domain_size_um
    axis.set(xlim=(-half, half), ylim=(-half, half), xlabel="x (µm)", ylabel="y (µm)")
    axis.set_aspect("equal")
    axis.set_title(
        f"Program19.3 · {config.equation_profile} · {config.parameter_profile} · "
        f"t = {state.time_s:.4g} s · {status}"
    )
    axis.grid(alpha=0.12)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def save_region_figure(
    path: Path,
    solid: np.ndarray,
    region: np.ndarray,
    extent: np.ndarray,
) -> None:
    plt = _pyplot()
    from matplotlib.colors import ListedColormap

    # 0 external, 1 void within analyzed region, 2 solid.
    classes = np.zeros_like(solid, dtype=np.uint8)
    classes[region] = 1
    classes[solid & region] = 2
    cmap = ListedColormap(("#e8edf0", "#49a9dc", "#d8c49b"))
    figure, axis = plt.subplots(figsize=(7.2, 7.2), dpi=180)
    axis.imshow(classes, origin="lower", extent=extent, cmap=cmap, vmin=0, vmax=2)
    axis.contour(
        region.astype(float),
        levels=(0.5,),
        origin="lower",
        extent=extent,
        colors=("#e39b00",),
        linewidths=2.5,
    )
    axis.set(xlabel="x (µm)", ylabel="y (µm)", aspect="equal")
    axis.set_title("Analyzed clot region and excluded exterior")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
