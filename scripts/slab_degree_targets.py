"""Per-slab junction and platelet targets for the 3-D -> 2-D slab mapping.

Follows the stereology note: equal-volume cylinder R = 24 um, L_z = 32 um,
M slabs of thickness k = L_z / M, slab area A = pi R^2.
  platelets per slab  Q_A^+ * A = (E[H] + k) * N_V(platelet) * A, E[H] = 2.2 um
  junctions per slab  N_V(degree) * A * k   (no +E[H]: junctions are points)
Densities come from Program19Config.node_degree_density_um3 (capsule sample 1).

  python scripts/slab_degree_targets.py                     # print the table
  python scripts/slab_degree_targets.py --write-configs DIR # headless JSONs
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from program19_core.config import CAPSULE_SAMPLE1_VOLUME_UM3  # noqa: E402
from program19_core.profiles import make_profiled_config  # noqa: E402
from program19_core.geometry import junction_degree_targets  # noqa: E402

R_UM, LZ_UM, MEAN_CALIPER_UM = 24.0, 32.0, 2.2
PLATELET_NV_UM3 = 45 / CAPSULE_SAMPLE1_VOLUME_UM3


def slab_targets(M: int, densities: dict[int, float]) -> dict:
    k = LZ_UM / M
    area = math.pi * R_UM**2
    return {
        "M": M,
        "k_um": k,
        "platelets": (MEAN_CALIPER_UM + k) * PLATELET_NV_UM3 * area,
        "nodes": {d: int(round(nv * area * k)) for d, nv in sorted(densities.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--M", type=int, nargs="+", default=[3, 9, 15, 22])
    parser.add_argument("--write-configs", type=str, help="write one headless JSON per M here")
    args = parser.parse_args()
    base = make_profiled_config()
    densities = {d: base.node_degree_density_um3[d] for d in base.node_degrees}
    degrees = sorted(densities)
    print("  M    k_um  platelets  " + "  ".join(f"deg{d:<2d}" for d in degrees) + "  total")
    for M in args.M:
        row = slab_targets(M, densities)
        nodes = row["nodes"]
        print(
            f"{M:3d}  {row['k_um']:6.2f}  {row['platelets']:6.2f} ({round(row['platelets']):2d})  "
            + "  ".join(f"{nodes[d]:5d}" for d in degrees)
            + f"  {sum(nodes.values()):5d}"
        )
        if args.write_configs:
            out = Path(args.write_configs)
            out.mkdir(parents=True, exist_ok=True)
            config = make_profiled_config(
                overrides=dict(
                    platelet_count=int(round(row["platelets"])),
                    effective_thickness_um=row["k_um"],
                    layer_spacing_um=row["k_um"],
                    mechanical_layers=M,
                )
            )
            # The density path (default domain 48 um, R = 24 um) gives the same counts.
            assert junction_degree_targets(config) == nodes, (M, junction_degree_targets(config))
            config.save_json(out / f"slab_M{M}.json")


if __name__ == "__main__":
    main()
