"""
Generate the Tasker III reconstructions of CeO2 (001) with
generate_slabs_for_miller and candidates="all".

CeO2 (001) is polar: half of the surface O (or Ce) atoms have to move from
one side of the slab to the other.  Each returned termination is one
symmetry-distinct way of choosing them; ``multiplicity`` counts the
equivalent patterns it stands for.  They are ranked by broken bonds, then
by how evenly the remaining surface atoms are spread.

Headless by default: writes structures under example/output_tasker3/.
Pass --plot for stacking-axis PNG plots, --verbose for the full analysis,
--view to open ASE's GUI after generation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ase.io import read, write

from taskerslabgen import generate_slabs_for_miller, plane_name_for_filename


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--view",
        action="store_true",
        help="Open ASE GUI after writing structures (interactive).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Write stacking-axis PNG plots next to the structures.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the plane, stacking and reconstruction analysis.",
    )
    args = parser.parse_args(argv)

    here = Path(__file__).resolve().parent
    bulk_path = here / ".." / "bulk_files" / "CeO2_fluorite.cif"
    charges = {"Ce": 4.0, "O": -2.0}
    miller = (0, 0, 1)

    output_dir = here / "output_tasker3"
    output_dir.mkdir(parents=True, exist_ok=True)

    bulk = read(bulk_path.as_posix())

    result = generate_slabs_for_miller(
        bulk,
        charges,
        millers=miller,
        layer_thickness_list=[2],
        bulk_name="CeO2",
        vacuum=15.0,
        plot=args.plot,
        plot_out_dir=output_dir.as_posix(),
        verbose=args.verbose,
        # Count only Ce-O bonds (covalent radii would also count Ce-Ce).
        bond_distances={"Ce-Ce": None, "O-O": None, "Ce-O": 2.35},
        candidates="all",
        # Ce- and O-terminated reconstructions break the same number of
        # bonds; keep the O-terminated ones, the termination usually
        # modelled for CeO2 (001).
        prefer_plane="O",
    )

    slabs = []
    for miller_key, terminations in result.items():
        print(f"\nMiller {miller_key}: {len(terminations)} termination(s)")
        for tid, info in terminations.items():
            slab = info["atoms"][0]
            slabs.append(slab)
            fname = f"CeO2_hkl_001_term_{tid}_{plane_name_for_filename(info['plane_type'])}.cif"
            out_path = output_dir / fname
            write(out_path.as_posix(), slab)
            cand = info["candidate"]
            print(
                f"  ID {tid}: type={info['tasker_type']}  "
                f"plane={info['plane_type']}  atoms={len(slab)}  "
                f"broken bonds={cand['bond_score']}  "
                f"multiplicity={cand['multiplicity']}  -> {fname}"
            )

    if not slabs:
        print("No slabs generated.")
        return 1

    if args.view:
        from ase.visualize import view

        view(slabs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
