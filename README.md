# platelet-fibrin-2d-core

Minimal, self-contained 2-D platelet-fibrin clot mechanics core (Program19.3).
Generates a 2-D planar clot slab and runs its overdamped mechanics on the CPU
(`--backend python`) or an NVIDIA GPU (`--backend cuda`).

* **Network generation** (`geometry.py`): platelets, junctions of degree 3–7 from the
  Michael & Pancaldi (2023) Code Ocean capsule sample 1 densities (`R = 24 um`,
  domain `48 um`), planar fiber pairing, and fiber subdivision.
* **Mechanics** (`cuda_mechanics.py` + `mechanics_kernel.cu` on GPU; `wlc.py`,
  `bending.py`, `cohesion.py`, `interactions.py`, `integrator.py`, `spatial.py` on CPU):
  Marko–Siggia WLC fibers (`N = 1100`), bending at interior and junction triples,
  permanent zero-at-birth fiber–fiber cohesion, platelet–fibrin and platelet–platelet
  contact forces, filopodia attachment/pulling, and adaptive Euler–Maruyama integration
  with Brownian noise.
* **Outputs**: `initial_state.npz`, `final_state.npz`, `trajectory.npz` (positions and
  cohesion/filopodia topology at each `output_interval_s`), `diagnostics.csv`,
  `layer_summary.json`, `layer_index.csv`, `run_metrics.json`, `manifest.json`, and
  `status.json`.

With the same config and seed, `initial_state.npz`, `final_state.npz`, and
`trajectory.npz` are bitwise identical to `platelet-fibrin-transport-lab-modernized`.

## Requirements

* `numpy` (always required)
* `scipy` (optional; used by `spatial.py` for KDTree neighbor search in `--backend python`)
* `nvidia-cuda-nvrtc-cu12` + NVIDIA driver on Windows (for `--backend cuda`)

```bash
python -m pip install -r requirements.txt
```

## Quickstart

```bash
python scripts/slab_degree_targets.py --write-configs configs
python src/program19_3.py --config configs/slab_M3.json --layers 1 --duration 60 --backend cuda --output-root runs
```

Useful options: `--config FILE`, `--duration S`, `--dt S`, `--seed INT`, `--platelets INT`,
`--layers N`, `--layer-seed-stride INT`, `--backend {python,cuda}`,
`--equation-profile {paper_consistent,codeocean_literal}`, `--output-root DIR`,
`--self-test`, `--write-default-config FILE`.
