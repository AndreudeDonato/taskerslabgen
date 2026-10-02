"""
Tandem genslab + cutslab workflow for CeO2 fluorite supercell.

For each Miller index:
  1. genslab generates a thick reference slab with the best O-terminated plane.
  2. cutslab cuts the thick slab into thinner sub-slabs preserving the
     same termination (including Tasker III reconstruction if applicable).
  3. Each sub-slab is saved with the naming convention:

     {stem}_hkl_{millerindex}_between_{bottom}_{top}_cut_{cutindex}.cif

Headless by default. Pass --plot to write PNG stacking plots.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ase.io import read, write

from _requires import require_taskerslabgen

require_taskerslabgen()  # clear message if Python imports an older taskerslabgen

from taskerslabgen import cutslab, generate_slabs_for_miller, plane_name_for_filename


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Write stacking-axis PNG plots next to the structures.",
    )
    args = parser.parse_args(argv)

    here = Path(__file__).resolve().parent
    bulk_path = here / ".." / "bulk_files" / "CeO2_fluorite_supercell2x2x2.cif"
    charges = {"Ce": 4.0, "O": -2.0}
    millers = [(1, 1, 0), (0, 0, 1)]
    ext = "cif"

    output_dir = here / "output_x2supercell"
    output_dir.mkdir(parents=True, exist_ok=True)

    bulk = read(bulk_path.as_posix())
    stem = bulk_path.stem

    # (001): keep the O-terminated reconstruction (see CeO2_fluorite.py).
    plane = [None, "O"]
    for i, miller in enumerate(millers):
        hkl_str = "".join(str(x) for x in miller)
        print("=" * 60)
        print(f"Miller {miller}")
        print("=" * 60)

        genslab_result = generate_slabs_for_miller(
            bulk_atoms=bulk,
            charges=charges,
            millers=miller,
            layer_thickness_list=[3],
            bulk_name=stem,
            vacuum=15.0,
            plot=args.plot,
            plot_out_dir=output_dir.as_posix(),
            # Count only Ce-O bonds (covalent radii would also count Ce-Ce).
            bond_distances={"Ce-Ce": None, "O-O": None, "Ce-O": 2.35},
            prefer_plane=plane[i],
            candidates="best",
        )

        terminations = genslab_result[miller]
        if not terminations:
            print(f"  No termination found for {miller}, skipping.\n")
            continue

        tid = min(terminations.keys())
        term = terminations[tid]
        thick_slab = term["atoms"][0]
        print(
            f"\n  Thick slab: {len(thick_slab)} atoms, "
            f"Tasker {term['tasker_type']}, plane={term['plane_type']}"
        )

        print(f"\n  Cutting thick slab for {miller}...")
        sub_slabs = cutslab(
            input_structure=thick_slab,
            charges=charges,
            axis=2,
            plot=args.plot,
            plot_out_dir=output_dir.as_posix(),
            cut_at="termination",
            reconstruction=term.get("reconstruction"),
            vacuum=15.0,
            cuts="right",
        )

        print(f"\n  Generated {len(sub_slabs)} sub-slabs for {miller}")
        for cut_i, slab in enumerate(sub_slabs):
            bp = plane_name_for_filename(slab.info.get("cut_bottom_plane", "?"))
            tp = plane_name_for_filename(slab.info.get("cut_top_plane", "?"))
            fname = (
                f"{stem}_hkl_{hkl_str}_between_{bp}_{tp}_cut_{cut_i}.{ext}"
            )
            out_path = output_dir / fname
            write(out_path.as_posix(), slab)
            print(f"    saved: {fname}  ({len(slab)} atoms)")

        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
