"""
Cut a relaxed slab into a thickness series with cutslab(bulk_atoms=...).

Relaxation moves the surface planes: they rumple, shift towards the bulk
and stop matching the bulk planes exactly.  Passing the bulk lets cutslab
assign every atom to the nearest bulk plane, so relaxed planes stay whole,
and label each plane by its bulk plane (``~``, e.g. ``O4~``, marks a plane
deformed by more than ``deform_tol``).

By default the script fakes a relaxation of IrO2 (110) from the shipped CIF
(surface atoms pulled towards the slab centre).  To cut your own relaxed
slab, pass it with the bulk it was built from (any ASE-readable files, e.g.
FHI-aims outputs; the bulk may be a supercell):

    python example/relaxed_cutslab.py --slab slab.out --bulk bulk.out --miller 1 1 0

Sub-slabs, and a plot of every cut, are written under example/output_relaxed/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ase.io import read, write

from _requires import require_taskerslabgen

require_taskerslabgen()  # clear message if Python imports an older taskerslabgen

from taskerslabgen import cutslab, generate_slabs_for_miller


def fake_relaxed_slab(bulk, charges, miller, depth=2.5, shift=0.15):
    """A genslab slab whose atoms within *depth* of a surface move inwards."""
    result = generate_slabs_for_miller(
        bulk, charges, miller, layer_thickness_list=[6], vacuum=10.0
    )
    slab = result[miller][0]["atoms"][0]
    z = slab.positions[:, 2]
    top, bottom = z.max(), z.min()
    pos = slab.get_positions()
    pos[z > top - depth, 2] -= shift
    pos[z < bottom + depth, 2] += shift
    slab.set_positions(pos)
    return slab


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slab", type=Path, help="Relaxed slab file.")
    parser.add_argument("--bulk", type=Path, help="Bulk the slab was built from.")
    parser.add_argument(
        "--miller", type=int, nargs=3, default=(1, 1, 0), help="Miller index."
    )
    parser.add_argument(
        "--dipole-tol",
        type=float,
        default=0.05,
        help="polarity (dipole per surface area, charges normalised, 1/A) still "
        "counted as zero; sub-slabs of relaxed slabs keep one relaxed surface "
        "and need more than the 1e-3 default.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the plots of where the slab was cut.",
    )
    args = parser.parse_args(argv)

    here = Path(__file__).resolve().parent
    miller = tuple(args.miller)
    output_dir = here / "output_relaxed"
    output_dir.mkdir(parents=True, exist_ok=True)

    if (args.slab is None) != (args.bulk is None):
        parser.error("--slab and --bulk go together")

    if args.slab is None:
        bulk = read((here / ".." / "bulk_files" / "IrO2_rutile.cif").as_posix())
        charges = {"Ir": 4.0, "O": -2.0}
        slab = fake_relaxed_slab(bulk, charges, miller)
        stem = "IrO2_rutile"
    else:
        bulk = read(args.bulk.as_posix())
        slab = read(args.slab.as_posix())
        # Formal charges +4/-2 suit the dioxides; edit for other materials.
        charges = {s: -2.0 if s == "O" else 4.0 for s in set(slab.get_chemical_symbols())}
        stem = args.slab.stem

    sub_slabs = cutslab(
        slab,
        charges,
        bulk_atoms=bulk,
        miller=miller,
        dipole_tol=args.dipole_tol,
        cut_at="termination",
        cuts="top",
        vacuum=15.0,
        plot=not args.no_plot,
        plot_out_dir=output_dir.as_posix(),
    )

    hkl = "".join(str(i) for i in miller)
    print(f"{stem} {miller}: {len(slab)} atoms -> {len(sub_slabs)} sub-slabs")
    for i, sub in enumerate(sub_slabs):
        bottom, top = sub.info["cut_bottom_plane"], sub.info["cut_top_plane"]
        fname = f"{stem}_hkl_{hkl}_cut_{i}.cif"
        write((output_dir / fname).as_posix(), sub)
        print(
            f"  {len(sub):4d} atoms  {sub.info['cut_n_planes']:3d} planes  "
            f"bottom={bottom:8s} top={top:8s} -> {fname}"
        )
    if not sub_slabs:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
