"""
Generate every Tasker I/II termination of IrO2 for several Miller indices.

Terminations come ranked: ID 0 breaks the fewest bonds (then is the densest
surface), which is what ``candidates="best"`` returns.

Writes structures and a plot of every termination (where the bulk was
cut) under example/output_tasker2/.  Pass --no-plot to skip the plots, --verbose for the full analysis, --view to open
ASE's GUI after generation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ase.io import read, write

from _requires import require_taskerslabgen

require_taskerslabgen()  # clear message if Python imports an older taskerslabgen

from taskerslabgen import generate_slabs_for_miller, plane_name_for_filename


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--view",
        action="store_true",
        help="Open ASE GUI after writing structures (interactive).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the plots of where the slabs were cut.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the plane and stacking analysis.",
    )
    args = parser.parse_args(argv)

    here = Path(__file__).resolve().parent
    bulk_path = here / ".." / "bulk_files" / "IrO2_rutile.cif"
    charges = {"Ir": 4.0, "O": -2.0}

    millers = [
        (1, 1, 1),
        (0, 0, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
    ]

    output_dir = here / "output_tasker2"
    output_dir.mkdir(parents=True, exist_ok=True)

    bulk = read(bulk_path.as_posix())
    stem = bulk_path.stem

    result = generate_slabs_for_miller(
        bulk_atoms=bulk,
        charges=charges,
        millers=millers,
        layer_thickness_list=[2],
        bulk_name=stem,
        vacuum=15.0,
        plot_out_dir=output_dir.as_posix(),
        verbose=args.verbose,
        plot=not args.no_plot,
        candidates="all",
    )

    slabs = []
    for miller in millers:
        terminations = result[miller]
        if not terminations:
            print(f"  No termination found for {miller}")
            continue
        print(f"  {miller}: {len(terminations)} termination(s)")
        hkl = "".join(str(i) for i in miller)
        for tid, info in terminations.items():
            slab = info["atoms"][0]
            fname = f"{stem}_hkl_{hkl}_term_{tid}_{plane_name_for_filename(info['plane_type'])}.cif"
            write((output_dir / fname).as_posix(), slab)
            bonds = info["candidate"]["broken_bonds_by_pair"]
            print(
                f"    ID {tid}: {len(slab)} atoms, Tasker {info['tasker_type']}  "
                f"plane={info['plane_type']}  broken bonds={bonds}  -> {fname}"
            )
            slabs.append(slab)

    if not slabs:
        print("No slabs generated.")
        return 1

    if args.view:
        from ase.visualize import view

        view(slabs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
