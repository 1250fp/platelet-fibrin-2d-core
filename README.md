# platelet-fibrin-2d-core

A lean copy of Program19.3 that does one thing: generate a 2-D platelet-fibrin clot
slab and run its overdamped mechanics, headless, on the CPU or an NVIDIA GPU.

* Network generation: platelets, junctions of degree 3-7 from the Michael & Pancaldi (2023)
  Code Ocean capsule sample 1 densities (`node_degree_model="density_table"`, R = 24 um,
  domain 48 um), planar fiber pairing, fiber subdivision.
* Mechanics: WLC fibers (N = 1100), bending, fiber-fiber cohesion, platelet contacts,
  filopodia pulling, adaptive overdamped integrator with Brownian noise.
* Independent z layers (optionally in parallel processes), checkpoints, diagnostics,
  trajectory, optional raster masks and PNG figures.

Built from `platelet-fibrin-transport-lab-modernized` (2026-10-22). With the same
config and seed, every simulation output (state, trajectory, checkpoint and mask arrays,
diagnostics) is bitwise identical to that tree's.

## Requirements

Python 3.11 or newer (validated on 3.14) and:

| Package | Needed for |
|---------|------------|
| numpy, scipy | always |
| matplotlib | figures and raster masks (optional: `--no-figures --no-masks`) |
| threadpoolctl | optional, limits BLAS threads in `--workers` processes |
| nvidia-cuda-nvrtc-cu12 + NVIDIA driver | optional, `--backend cuda` (Windows only) |

```
python -m pip install -r requirements.txt
```

## Quickstart (from this folder)

```
python scripts/slab_degree_targets.py --write-configs configs
python src/program19_3.py --headless --config configs/slab_M3.json --layers 1 --duration 60 --output-root runs
```

The first command prints the slab table (M = 3, 9, 15, 22: thickness, platelets and
junction targets for degrees 3-7) and writes `configs/slab_M<M>.json`. The second
runs one layer of the M = 3 slab for 60 s of mechanics.

Useful options: `--seed`, `--dt`, `--layers N`, `--workers N` (parallel layers),
`--backend cuda`, `--no-figures`, `--no-masks`, `--no-intermediate-masks`,
`--parameter-profile`, `--equation-profile`, `--numerical-profile`, `--self-test`,
`--write-default-config FILE`, `--write-profile-guide FILE`.
Without `--output-root`, results go to `src/results/`.

## Outputs

One timestamped run folder:

| File | Contents |
|------|----------|
| `config.json`, `profile_audit.json`, `manifest.json` | full configuration, deviations from named profiles, provenance (equations, versions, source hashes) |
| `layer_index.csv`, `run_metrics.json`, `status.json` | per-layer summary, run summary, final status |
| `mechanical_spatial_stack.npz` | final solid/region masks of all layers (z, y, x) - masks only |
| `mechanical_layers/layer_NNN_seedS/initial_state.npz`, `final_state.npz` | complete network state at t = 0 and at the end |
| `.../trajectory.npz` | positions (and cohesion/filopodia topology) every `output_interval_s` |
| `.../checkpoint.npz` | state + RNG for restarting |
| `.../diagnostics.csv` | forces, strain, cohesion links, substeps per saved step |
| `.../layer_summary.json` | per-layer counts and status |
| `.../final_masks.npz`, `intermediate_masks.npz` | rasterized solid / analyzed region - masks only |
| `.../final_mechanics.png`, `analyzed_region_2d.png` | figures - figures only |

## CPU vs GPU backend

`--backend python` (default) is the reference. `--backend cuda` compiles
`mechanics_kernel.cu` at run time with NVRTC. It needs Windows (it loads
`nvcuda.dll`), an NVIDIA GPU and `nvidia-cuda-nvrtc-cu12`. On the GPU, each
outer step runs the adaptive substeps there: WLC, bending, platelet contacts,
platelet-platelet forces, filopodia attachment and pulling, Brownian noise, and
the moves. The cohesion candidate search and acceptance run there too.
Network generation, cohesion degree counting, checkpoints, masks and figures stay on
the CPU, and positions are copied back after every outer step. Arithmetic is float64,
compiled with `--fmad=false`.

Without Brownian motion the GPU matches the CPU to about 1e-8 um (the repository's tests
use rtol = atol = 1e-8), not bitwise: force sums use atomic adds whose order varies.
With Brownian motion the trajectories differ from the CPU run. The GPU draws its noise
from a Philox stream seeded by `seed`, not from NumPy, so its results agree with the
CPU only in distribution. The GPU path supports only `gaussian_independent` noise and
`nearest_free` / `global_reacquire` filopodia. It reports `filopodia_rms_nN` as 0.

## What was removed (still in `platelet-fibrin-transport-lab-modernized`)

| Removed | Was |
|---------|-----|
| `transport.py`, `transport_steps.py` | pore classification and accessibility (EDT), first-passage transport |
| pore outputs | `pore_analysis.npz`, `pore_summary.json`, `mechanical_stack_and_pore_slices.png`, and the porosity / MET / escape metrics in `run_metrics.json` |
| `codeocean_geometry.py` | CodeOcean XML projection geometry (`geometry_profile="codeocean_xml_xy"`, `--codeocean-xml`) |
| `gui.py` | Tk GUI (running without `--headless`) |
| `continuation.py`, `stack_extension.py`, `catalog.py` | continuing runs, adding layers, run catalogue |
| `evolving_transport.py`, `through_clot.py`, `pilot.py`, `experiments.py` | transport analyses, pilots, f0 sweeps |
| `runner.py` | alias module |
| `src/continuous_transport/` | C++ transport engines |
| `tests/`, `validation/`, `paper/`, `data/`, `results/`, `archive/`, `docs/` | tests, validation, manuscript, data, past runs |

The removed command-line options (`--transport`, `--continue-run`, `--extend-stack`,
`--f0-sweep`, `--catalog-root`, the transport-analysis options, `--codeocean-xml`,
`--geometry-profile`) now stop with a message that points to the modernized folder.
The transport fields in `config.json` (`particle_radius_um`, `run_transport`, ...) are
kept so configs stay compatible with the full tree. `run_transport=true` is rejected.
