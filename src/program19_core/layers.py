"""Rasterisation of explicit mechanical platelet--fibrin layers."""

from __future__ import annotations

import numpy as np

try:
    from scipy.ndimage import binary_dilation
    from scipy.spatial import ConvexHull
except ImportError as exc:  # pragma: no cover
    binary_dilation = None
    ConvexHull = None
    SCIPY_ERROR = exc
else:
    SCIPY_ERROR = None

from .config import Program19Config
from .state import SimulationState


def analyzed_region(config: Program19Config) -> tuple[np.ndarray, np.ndarray, float]:
    resolution = int(config.raster_resolution)
    half = 0.5 * config.domain_size_um
    dx = config.domain_size_um / resolution
    coordinates = -half + (np.arange(resolution) + 0.5) * dx
    x, y = np.meshgrid(coordinates, coordinates)
    region = x * x + y * y <= half * half
    extent = np.asarray((-half, half, -half, half), dtype=float)
    return region, extent, dx


def _paint_segment(
    mask: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    radius: float,
    half: float,
    dx: float,
) -> None:
    resolution = mask.shape[0]
    xmin, xmax = min(a[0], b[0]) - radius, max(a[0], b[0]) + radius
    ymin, ymax = min(a[1], b[1]) - radius, max(a[1], b[1]) + radius
    ix0 = max(0, int(np.floor((xmin + half) / dx)))
    ix1 = min(resolution, int(np.ceil((xmax + half) / dx)) + 1)
    iy0 = max(0, int(np.floor((ymin + half) / dx)))
    iy1 = min(resolution, int(np.ceil((ymax + half) / dx)) + 1)
    if ix0 >= ix1 or iy0 >= iy1:
        return
    xs = -half + (np.arange(ix0, ix1) + 0.5) * dx
    ys = -half + (np.arange(iy0, iy1) + 0.5) * dx
    x, y = np.meshgrid(xs, ys)
    delta = b - a
    length2 = float(np.dot(delta, delta))
    if length2 <= 1e-18:
        distance2 = (x - a[0]) ** 2 + (y - a[1]) ** 2
    else:
        t = np.clip(((x - a[0]) * delta[0] + (y - a[1]) * delta[1]) / length2, 0.0, 1.0)
        distance2 = (x - (a[0] + t * delta[0])) ** 2 + (y - (a[1] + t * delta[1])) ** 2
    mask[iy0:iy1, ix0:ix1] |= distance2 <= radius**2


def rasterize_geometry(
    nodes_um: np.ndarray,
    platelets_um: np.ndarray,
    backbone_pairs: np.ndarray,
    config: Program19Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Rasterize one explicit mechanical state and its analyzed support."""
    region, extent, dx = analyzed_region(config)
    solid = np.zeros_like(region)
    half = 0.5 * config.domain_size_um
    nodes = np.asarray(nodes_um, dtype=float)
    for a, b in np.asarray(backbone_pairs, dtype=int).reshape((-1, 2)):
        _paint_segment(
            solid,
            nodes[int(a)],
            nodes[int(b)],
            config.fiber_radius_um,
            half,
            dx,
        )
    coordinates = -half + (np.arange(config.raster_resolution) + 0.5) * dx
    x, y = np.meshgrid(coordinates, coordinates)
    for center in np.asarray(platelets_um, dtype=float).reshape((-1, 2)):
        solid |= (x - center[0]) ** 2 + (y - center[1]) ** 2 <= config.platelet_radius_um**2
    solid &= region
    # The analyzed clot region is the padded convex support of the actual solid
    # phase, not the entire initial disk. This prevents empty exterior from
    # being counted as clot porosity. The initial disk remains a hard domain cap.
    if ConvexHull is not None and np.count_nonzero(solid) >= 3:
        from matplotlib.path import Path as PolygonPath

        pixels_yx = np.argwhere(solid)
        points_xy = pixels_yx[:, ::-1].astype(float)
        try:
            hull = ConvexHull(points_xy)
            polygon = PolygonPath(points_xy[hull.vertices])
            yy, xx = np.indices(solid.shape)
            support = polygon.contains_points(
                np.column_stack((xx.ravel(), yy.ravel())), radius=1e-9
            ).reshape(solid.shape)
            padding_px = int(np.ceil(config.clot_support_padding_um / dx))
            if padding_px > 0:
                support = binary_dilation(support, iterations=padding_px)
            region = (support | solid) & region
        except Exception:
            # Degenerate hulls keep the circular domain; the manifest still
            # makes this fallback inspectable through the saved region mask.
            pass
    return solid, region, extent, dx


def rasterize_solid(
    state: SimulationState, config: Program19Config
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    return rasterize_geometry(
        state.nodes_um,
        state.platelets_um,
        state.backbone_pairs,
        config,
    )
