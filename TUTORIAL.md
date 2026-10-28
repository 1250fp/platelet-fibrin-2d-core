# Tutorial: running a full M-slab clot and reading the outputs

This guide covers generating the slab configs, running a whole clot (or a single
test layer), and loading the results with numpy. Every command and flag below
exists in `src/program19_core/cli.py` or `scripts/slab_degree_targets.py`. The
file names and array keys come from a real run.

Units throughout: length µm, force nN, time s.

---

## 1. What the core simulates

**One run of the core simulates one clot as a stack of independent 2-D slabs.**
The reference clot is an equal-volume cylinder of radius **R = 24 µm** and
height **L_z = 32 µm**. It is cut into **M slabs** of thickness **k = L_z / M**.
Each slab is collapsed into a 2-D disc of radius R, which gives a domain of
48 µm (`domain_size_um`), and is simulated as its own mechanical layer.
Layers don't exchange forces.

Each layer is generated from its own seed (see §4) and contains:

* **Fibrin junctions** (branch points) of degree 3–7, connected by planar,
  non-crossing fibres. Each fibre is subdivided into ≈0.30 µm segments
  (`segment_target_um`), so the final node count is much larger than the
  junction count (about 2,400 nodes for an M = 16 slab).
* **Platelets** (discs of radius 1.13 µm) with filopodia that attach to fibrin
  nodes and pull on them.
* The mechanics: WLC fibre stretching, bending, platelet–fibrin and
  platelet–platelet contact, permanent fibre–fibre cohesion, filopodia pulling,
  and overdamped Brownian motion (adaptive Euler–Maruyama).

### Stereology: how many junctions and platelets per slab

`scripts/slab_degree_targets.py` converts the 3-D densities of the Michael &
Pancaldi (2023) capsule, sample 1, into per-slab targets. The slab area is
A = πR²:

* **Junctions of degree d per slab:** `N_V(d) · A · k`, rounded. Junctions are
  points, so there is no extra caliper term. `N_V(d)` is the sample-1 count of
  junctions of degree d (416, 289, 102, 95 and 80 for d = 3…7) divided by the
  sphere volume 4/3·π·24³ ≈ 57,906 µm³.
* **Platelets per slab:** `(E[H] + k) · N_V(platelet) · A`, rounded, with mean
  caliper diameter E[H] = 2.2 µm and N_V(platelet) = 45 / 57,906 µm⁻³. The
  `+E[H]` term counts platelets whose body cuts the slab even though their
  centre lies outside it. That is why thin slabs still have several platelets.

Output of `python scripts/slab_degree_targets.py --M 4 8 16`:

```
  M    k_um  platelets  deg3   deg4   deg5   deg6   deg7   total
  4    8.00   14.34 (14)    104     72     26     24     20    246
  8    4.00    8.72 ( 9)     52     36     13     12     10    123
 16    2.00    5.91 ( 6)     26     18      6      6      5     61
```

("total" counts junctions only, not the subdivision nodes.)

Inside the core, `geometry.junction_degree_targets()` applies the same formula
using `domain_size_um / 2` as R and `effective_thickness_um` as k. An explicit
`node_degree_counts` entry in the config overrides the density for that degree.

---

## 2. Install

```bash
python -m pip install -r requirements.txt
```

* `numpy` is required. `scipy` is optional and speeds up neighbour search for
  `--backend python`. Both are pinned in `requirements.txt` (validated on
  CPython 3.14).
* **CUDA backend (Windows + NVIDIA GPU only):** the GPU path loads `nvcuda.dll`
  from the NVIDIA driver and compiles the kernel at run time with NVRTC from
  pip:

  ```powershell
  py -3 -m pip install nvidia-cuda-nvrtc-cu12
  ```

  No CUDA toolkit install is needed. On Linux/macOS use `--backend python`.
* On Windows, `py` works anywhere this guide says `python`.

Quick check:

```bash
python src/program19_3.py --self-test
```

---

## 3. Generate the slab configs

```bash
python scripts/slab_degree_targets.py --M 4 8 16 --write-configs configs
```

This prints the table above and writes `configs/slab_M4.json`,
`configs/slab_M8.json` and `configs/slab_M16.json`. Without `--M`, it uses
M = 3, 9, 15, 22. Each JSON is a complete `Program19Config` dump: every
parameter in `src/program19_core/config.py`, with defaults, except for four
fields set per M:

| field | meaning | M = 16 value |
|---|---|---|
| `platelet_count` | platelets per slab (rounded stereology value) | 6 |
| `effective_thickness_um` | slab thickness k, used for the junction counts | 2.0 |
| `layer_spacing_um` | z spacing between layers, used for `z_um` in the outputs | 2.0 |
| `mechanical_layers` | number of slabs M (one layer per slab) | 16 |

Other fields worth knowing (edit them in the JSON; most have no CLI flag):

| field | default | notes |
|---|---|---|
| `duration_s` | 1.0 | also `--duration` |
| `dt_s` | 0.001 | outer step; also `--dt`. Adaptive substeps run inside it |
| `output_interval_s` | 0.10 | **trajectory frame spacing, JSON only** (see the warning in §5) |
| `seed` | 2035 | base seed; also `--seed` |
| `mechanics_backend` | `"python"` | also `--backend` |
| `filopodia_f0_nN` | 2.233 | **filopodia pulling force f0**, JSON only |
| `trajectory_detail` | `"topology"` | `"positions"` drops the cohesion and filopodia arrays from `trajectory.npz` |
| `save_trajectory` | true | |
| `effective_node_drag_nN_s_per_um` / `effective_platelet_drag_nN_s_per_um` | 3.77 / 85.2 | see §7 |

### Setting the pulling force f0

`filopodia_f0_nN` is the f0 term of the per-arm filopodial force law in
`interactions.filopodial_force_nN` (the same law is used on the GPU):

```
F = max(0, 1/2 * (f0 + a*k / (b*k + c)))     # k = local fibre stiffness (nN)
```

`a`, `b` and `c` are `filopodia_a/b/c`, set to 61.0, 3.582 and 131.8 by
default. With **f0 = 0, the arms still pull** through the stiffness term. To
switch pulling off entirely you need different settings, for example
`filopodia_count_model = "fixed"` with `filopodia_fixed_count = 0` (checked on
the CPU backend).

The CLI has no f0 flag, so make one JSON per force. Example for f0 = 0, 2.2 and
4.4 nN:

```python
import json
base = json.load(open("configs/slab_M16.json"))
for f0 in (0.0, 2.2, 4.4):
    cfg = dict(base, filopodia_f0_nN=f0, output_interval_s=30.0)
    json.dump(cfg, open(f"configs/slab_M16_f{f0}.json", "w"), indent=2)
```

Unknown keys in a JSON are ignored silently, so check the spelling against
`config.py`. Invalid values (for example a negative duration or an unknown
backend) raise an error at load time.

---

## 4. Run a clot

### A whole clot (all M layers)

```bash
python src/program19_3.py --config configs/slab_M16.json --seed 5200 --duration 1800 --backend cuda --output-root runs
```

This runs layers 1…16 one after another and prints a single-line progress
display (`% layer=i/M t=… links=… substeps=…`). Command-line flags override
the JSON values. Without `--seed`, the config's `seed` is used (2035 in the
generated configs).

### One layer, for testing

```bash
python src/program19_3.py --config configs/slab_M16.json --layers 1 --duration 1 --backend cuda --output-root runs
```

`--layers N` runs only the first N layers (it overrides `mechanical_layers`).
On a machine without the GPU path, a CPU smoke test runs in a few seconds:

```bash
python src/program19_3.py --config configs/slab_M16.json --layers 1 --duration 0.05 --backend python --output-root runs
```

### All CLI flags

`--config FILE`, `--duration S`, `--dt S`, `--seed INT`, `--platelets INT`,
`--layers N`, `--layer-seed-stride INT`, `--backend {python,cuda}`,
`--equation-profile {paper_consistent,codeocean_literal}` (alias `--profile`),
`--output-root DIR`, `--self-test`, `--write-default-config FILE`. Any other
parameter has to be set in the JSON.

### Seeds: one per layer, and non-overlapping blocks across clots

Layer i (zero-based) uses

```
layer_seed = seed + i * layer_seed_stride      # layer_seed_stride defaults to 1
```

so `--seed 5200` with M = 16 uses seeds 5200…5215. The layer seed fixes the
geometry (junction and platelet placement, fibre pairing, filopodia counts)
and the CPU noise stream. **Two clots with the same config whose seed
ranges overlap share identical layers**, so give every clot or replica its own block:

```
replica r (0-based):  --seed  BASE + r * M * stride
e.g. M = 16, BASE = 5200:  5200, 5216, 5232, ...
```

To compare forces on identical geometry, reuse the **same** seed block for
each f0 config. Only the force then differs between the runs.

---

## 5. Outputs

Each run creates a new, timestamped folder under `--output-root` (default
`src/results/`):

```
runs/program19_3_16mechanical_layers_paper_consistent_seed5200_YYYYMMDD_HHMMSS/
├── manifest.json        program/version, full configuration, units, seed rule,
│                        equations, dependency versions, platform, SHA-256 of every source file
├── config.json          the effective config (JSON + CLI overrides)
├── layer_index.csv      one row per finished layer (= layer_summary fields + folder)
├── run_metrics.json     layers requested/completed, backend, filopodia_f0_nN
├── status.json          "completed"/"failed", layers saved, wall time, error
└── mechanical_layers/
    ├── layer_001_seed5200/
    │   ├── config.json          this layer's config (seed = layer seed)
    │   ├── initial_state.npz    full state at t = 0
    │   ├── final_state.npz      full state at the end
    │   ├── trajectory.npz       positions (+ topology) at every saved frame
    │   ├── diagnostics.csv      one row per saved frame
    │   └── layer_summary.json   seed, z_um, status, node/platelet/link counts, wall time
    ├── layer_002_seed5201/
    └── ...
```

### `trajectory.npz`

| key | shape | content |
|---|---|---|
| `times_s` | (F,) | frame times: t = 0, then one every `output_interval_s`, and the final time |
| `nodes_um` | (F, N, 2) | fibrin node positions |
| `platelets_um` | (F, P, 2) | platelet centres |
| `backbone_pairs` | (S, 2) | fibre segments (node index pairs); fixed in time |
| `fixed_nodes` | (N,) | pinned-node mask |
| `axis_order_for_positions` | () | the string `"time,object,xy"` |
| `cohesion_frame_offsets` | (F+1,) | links at frame f are rows `off[f]:off[f+1]` of the `*_flat` arrays |
| `cohesion_pairs_flat` | (ΣL, 2) | cohesive link node pairs, frame by frame |
| `cohesion_reference_lengths_um_flat` | (ΣL,) | rest length of each link (its length at birth) |
| `cohesion_birth_times_s_flat` | (ΣL,) | time each link formed |
| `filopodia_node_ids` | (F, P, A) | node held by each platelet's filopodium arm; −1 = free or no arm |

The last five keys exist only with `trajectory_detail = "topology"`, which is
the default.

### `initial_state.npz` / `final_state.npz`

`nodes_um`, `platelets_um`, `backbone_pairs`, `backbone_reference_lengths_um`,
`bending_triples`, `bending_rest_angles_rad`, `bending_dual_reference_lengths_um`,
`cohesion_pairs`, `cohesion_reference_lengths_um`, `cohesion_birth_times_s`,
`fixed_nodes`, `filopodia_node_ids`, `node_fiber_id_offsets`,
`node_fiber_ids_flat`, `time_s`, `step`.

Fibre IDs are stored ragged: the fibres through node n are
`node_fiber_ids_flat[off[n]:off[n+1]]`, with `off = node_fiber_id_offsets`.
Junction nodes belong to several fibres. Fibre IDs and the bending triples are
saved only here, not in `trajectory.npz`.

### `diagnostics.csv`

One row per saved frame after t = 0, with columns `step`, `simulated_time_s`, `wall_elapsed_s`,
force RMS values (`wlc_rms_nN`, `bending_rms_nN`, `contact_rms_nN`,
`filopodia_rms_nN`), `max_backbone_strain`, maximum node and platelet forces,
`net_internal_force_nN`, `attached_filopodia`, `platelet_overlap_count`,
`minimum_platelet_center_distance_um`, `new_cohesion_links`,
`total_cohesion_links`, `maximum_cohesion_degree`, maximum displacements,
`internal_substeps`, and node and platelet diffusion coefficients.

> **Warning: memory and `output_interval_s`.** Frames are kept in RAM and
> `trajectory.npz` is written only when the layer ends. With the default
> 0.1 s interval, a 1800 s run stores 18,001 frames, which is about 0.7 GB per
> layer at M = 16 and more at M = 4. For long runs, set `output_interval_s` to
> something like 1–30 s in the JSON. There is no checkpoint/resume in this
> core. A layer that is interrupted must be rerun. Finished layers keep their
> folders, and a new run can start from the next layer's seed with
> `--seed BASE + i*stride --layers M-i`. Note that its folder will be numbered
> from `layer_001`.

---

## 6. Loading and plotting a layer

This snippet plots fibres, cohesive links and platelets at the first and last
frames, plus the number of cohesive links over time. It has been tested on a
real run of this repo:

```python
import glob
import numpy as np
import matplotlib.pyplot as plt

run = sorted(glob.glob("runs/program19_3_*"))[-1]                 # newest run
layer = sorted(glob.glob(f"{run}/mechanical_layers/layer_*"))[0]  # first layer
traj = np.load(f"{layer}/trajectory.npz")

t = traj["times_s"]                 # (frames,)
nodes = traj["nodes_um"]            # (frames, nodes, 2)
plts = traj["platelets_um"]         # (frames, platelets, 2)
bb = traj["backbone_pairs"]         # (segments, 2) node indices, fixed topology
off = traj["cohesion_frame_offsets"]
links_per_frame = np.diff(off)      # cohesive links present at each frame

def links_at(f):                    # (n_links, 2) node pairs at frame f
    return traj["cohesion_pairs_flat"][off[f]:off[f + 1]]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, f in zip(axes[:2], (0, len(t) - 1)):
    x = nodes[f]
    segs = x[bb]                                          # (segments, 2, 2)
    ax.plot(segs[:, :, 0].T, segs[:, :, 1].T, color="0.4", lw=0.4)
    coh = x[links_at(f)]
    ax.plot(coh[:, :, 0].T, coh[:, :, 1].T, color="tab:red", lw=0.6)
    r = 1.13                                              # platelet_radius_um in config.json
    for cx, cy in plts[f]:
        ax.add_patch(plt.Circle((cx, cy), r, color="tab:blue", alpha=0.6))
    ax.set_aspect("equal")
    ax.set_title(f"t = {t[f]:g} s, {links_per_frame[f]} cohesive links")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
axes[2].plot(t, links_per_frame, "o-")
axes[2].set_xlabel("model time (s)")
axes[2].set_ylabel("cohesive links")
fig.tight_layout()
fig.savefig("layer_overview.png", dpi=150)
plt.show()
```

`matplotlib` isn't in `requirements.txt`; install it separately. To loop over
a whole clot, iterate over `mechanical_layers/layer_*` and use `z_um` from each
`layer_summary.json`, or `layer_index.csv`, to stack the slabs.

---

## 7. Notes

* **Time is model time.** The node and platelet drags
  (`effective_node_drag_nN_s_per_um = 3.77`,
  `effective_platelet_drag_nN_s_per_um = 85.2`, `mobility_model =
  "effective_network"`) are effective network values, roughly 10⁶× the Stokes
  drag of a free fibre or platelet in plasma. A uniform drag scale only
  rescales time, so "1800 s" is 1800 s of model time. Its mapping to real
  clot-contraction time still has to be calibrated against experimental
  kinetics. `mobility_model = "free_stokes"` switches to Stokes drag from
  `viscosity_Pa_s` for comparison.
* **Cohesive links are permanent.** Links form between nearby nodes of
  different fibres (`cohesion_distance_um` = 0.1 µm, checked every
  `cohesion_search_stride` outer steps, up to `cohesion_max_total_neighbors` per
  node) with zero force at birth (rest length = distance at formation). After that they
follow the same WLC law as the backbone, and they are never removed. Their count is
  non-decreasing, so the per-frame lists in `trajectory.npz` only grow.
* **Backends.**
  * `python` is the CPU reference (NumPy plus an optional SciPy KDTree). It
    runs on any OS and is slow for long runs.
  * `cuda` runs the adaptive substeps, cohesion search and filopodia updates on
    the GPU (Windows + NVIDIA). It is the backend for production runs.
  * Both build identical initial geometry from the same seed, so
    `initial_state.npz` matches. Their thermal noise comes from different
    generators: NumPy on the CPU and a Philox stream on the GPU. Their
    trajectories therefore agree statistically, not bitwise.
* **Reproducibility.**
  * The same config, seed and backend give the same results. Two CPU runs of
    the smoke test above produced bitwise-identical `trajectory.npz` files.
  * `manifest.json` records the full config, dependency versions, platform and
    SHA-256 hashes of every source file. Keep the run folder intact (or at
    least `manifest.json` + `config.json`) with any data you publish.
  * Record the git commit (`git rev-parse HEAD`) alongside your runs.
