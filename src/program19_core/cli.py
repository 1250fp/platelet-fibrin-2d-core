"""Command-line interface for the Program19.3 2-D clot mechanics core."""

from __future__ import annotations

import argparse
from collections.abc import Iterable

from .config import (
    EQUATION_PROFILES,
    NUMERICAL_PROFILES,
    PARAMETER_PROFILES,
    Program19Config,
)
from .profiles import make_profiled_config, profiles_markdown
from .mechanical_runner import run_pipeline

FULL_TREE = "platelet-fibrin-transport-lab-modernized"

# Options of the full Program19.3 tree that were removed from the 2-D core.
# They are still parsed so that using one fails with a clear message instead of
# a generic "unrecognized arguments" error.
REMOVED_OPTIONS = {
    "--transport": "first-passage tracer transport",
    "--injection-rate": "tracer injection (transport)",
    "--f0-sweep": "f0 sweep experiments",
    "--replicates": "f0 sweep experiments",
    "--catalog-root": "run catalogue",
    "--continue-run": "continuing a saved run",
    "--additional-time": "continuing a saved run",
    "--extend-stack": "stack extension",
    "--target-layers": "stack extension",
    "--first-new-seed": "stack extension",
    "--evolving-transport-run": "evolving transport analysis",
    "--frozen-transport-run": "frozen transport analysis",
    "--release-time": "transport analysis",
    "--observation-time": "transport analysis",
    "--particle-radius": "transport analysis",
    "--diffusion": "transport analysis",
    "--particles": "transport analysis",
    "--transport-seed": "transport analysis",
    "--recorded-particles": "transport analysis",
    "--no-frozen-comparison": "transport analysis",
    "--codeocean-xml": "CodeOcean XML projection geometry",
    "--geometry-profile": "CodeOcean XML projection geometry (only generated_2d is built)",
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Program19.3 2-D core: generated platelet-fibrin clot mechanics (headless)"
    )
    result.add_argument("--headless", action="store_true", help="run a simulation")
    result.add_argument("--config", type=str, help="JSON configuration file")
    result.add_argument("--duration", type=float)
    result.add_argument("--dt", type=float)
    result.add_argument("--seed", type=int)
    result.add_argument("--platelets", type=int)
    result.add_argument("--profile", dest="equation_profile", choices=EQUATION_PROFILES,
                        help="backward-compatible alias for --equation-profile")
    result.add_argument("--equation-profile", dest="equation_profile", choices=EQUATION_PROFILES)
    result.add_argument("--parameter-profile", choices=PARAMETER_PROFILES)
    result.add_argument("--numerical-profile", choices=NUMERICAL_PROFILES)
    result.add_argument("--output-root", type=str)
    result.add_argument("--layers", type=int, help="number of independent mechanical z layers")
    result.add_argument(
        "--workers", type=int,
        help="parallel layer processes (1 keeps serial execution)",
    )
    result.add_argument(
        "--backend", choices=("python", "cuda"),
        help="mechanical integrator; cuda runs each layer's adaptive substeps on the GPU",
    )
    result.add_argument("--layer-seed-stride", type=int)
    result.add_argument("--no-intermediate-masks", action="store_true")
    result.add_argument("--no-figures", action="store_true",
                        help="skip PNG figures (Matplotlib not needed if --no-masks too)")
    result.add_argument("--no-masks", action="store_true",
                        help="skip raster masks and the layer stack (they use matplotlib.path)")
    result.add_argument("--self-test", action="store_true")
    result.add_argument("--write-default-config", type=str)
    result.add_argument("--write-profile-guide", type=str)
    for option in REMOVED_OPTIONS:
        result.add_argument(option, nargs="*", default=None, help=argparse.SUPPRESS)
    return result


def _self_test() -> None:
    import numpy as np
    from .bending import bending_triplet
    from .wlc import paper_stiffness_dF_dgamma_nN, wlc_force_nN

    config = Program19Config(
        duration_s=0.002,
        dt_s=0.001,
        platelet_count=0,
        raster_resolution=32,
        mechanical_layers=3,
    )
    L0 = 0.3
    assert abs(wlc_force_nN(L0, L0, config)) < 1e-12
    assert wlc_force_nN(1.1 * L0, L0, config) > 0
    assert wlc_force_nN(0.9 * L0, L0, config) < 0
    gamma = 0.2
    eps = 1e-6
    numeric = (
        wlc_force_nN((1 + gamma + eps) * L0, L0, config)
        - wlc_force_nN((1 + gamma - eps) * L0, L0, config)
    ) / (2 * eps)
    assert np.isclose(numeric, paper_stiffness_dF_dgamma_nN(gamma, config), rtol=2e-5)
    xi, xj, xk = np.asarray((0.0, 0.0)), np.asarray((1.0, 0.0)), np.asarray((2.0, 0.0))
    forces = bending_triplet(xi, xj, xk, 0.0, 1.0, config)
    assert np.allclose(np.sum(forces, axis=0), 0.0)
    assert np.allclose(forces, 0.0)
    print("Program19.3 2-D core self-test passed")


def main(argv: Iterable[str] | None = None) -> int:
    args = parser().parse_args(argv)
    for option, feature in REMOVED_OPTIONS.items():
        if getattr(args, option.lstrip("-").replace("-", "_")) is not None:
            raise SystemExit(
                f"{option} ({feature}) is not part of the 2-D core; "
                f"use the {FULL_TREE} folder."
            )
    if args.self_test:
        _self_test()
        return 0
    if args.write_profile_guide:
        from pathlib import Path

        Path(args.write_profile_guide).write_text(profiles_markdown(), encoding="utf-8")
        print(f"Profile guide written to {args.write_profile_guide}")
        return 0
    if args.write_default_config:
        make_profiled_config(
            equation_profile=args.equation_profile or "paper_consistent",
            parameter_profile=args.parameter_profile or "thesis_audited",
            numerical_profile=args.numerical_profile or "corrected_adaptive",
        ).save_json(args.write_default_config)
        print(f"Default configuration written to {args.write_default_config}")
        return 0
    if not args.headless:
        raise SystemExit(
            f"The GUI is not part of the 2-D core (it lives in the {FULL_TREE} folder). "
            "Run with --headless, e.g. --headless --config configs/slab_M3.json"
        )
    config = (
        Program19Config.load_json(args.config)
        if args.config
        else make_profiled_config(
            equation_profile=args.equation_profile or "paper_consistent",
            parameter_profile=args.parameter_profile or "thesis_audited",
            numerical_profile=args.numerical_profile or "corrected_adaptive",
        )
    )
    values = config.to_dict()
    overrides = {
        "duration_s": args.duration,
        "dt_s": args.dt,
        "seed": args.seed,
        "platelet_count": args.platelets,
        "output_root": args.output_root,
        "mechanical_layers": args.layers,
        "mechanical_workers": args.workers,
        "layer_seed_stride": args.layer_seed_stride,
        "mechanics_backend": args.backend,
    }
    values.update({key: value for key, value in overrides.items() if value is not None})
    if args.no_intermediate_masks:
        values["save_intermediate_masks"] = False
    config = Program19Config.from_dict(values)

    def show(payload: dict) -> None:
        if payload.get("stage") == "mechanics":
            print(
                f"\r{100 * payload['fraction']:6.2f}%  "
                f"layer={payload.get('layer_index', 0)}/{payload.get('layer_count', 0)}  "
                f"t={payload.get('simulated_time_s', 0):.4g} s  "
                f"links={payload.get('cohesion_links', 0)}  "
                f"substeps={payload.get('internal_substeps', 1)}",
                end="",
                flush=True,
            )
        elif payload.get("stage") == "complete":
            print()

    _, directory, status = run_pipeline(
        config, progress_callback=show,
        figures=not args.no_figures, masks=not args.no_masks,
    )
    print(f"Status: {status}\nResults: {directory}")
    return 0
