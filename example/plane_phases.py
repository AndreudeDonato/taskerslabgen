"""
Plane phases: what the labels O, O', O'' mean and how selection= uses them.

Every plane is treated as a wave: a smooth periodic density with one Gaussian
per atom.  Two planes have the same *arrangement* when one wave overlaps the
other after some shift or rotation; they are in the same *phase* when they
overlap as they are.  A label is the arrangement plus its phase:

    O, O', O''      one arrangement of O atoms in three phases (shifted or rotated)
    IrO2-a, IrO2-b  two different arrangements of IrO2 (no shift or rotation maps them)

Three cases are shown, each with structures to look at in the ASE GUI:

1. Same arrangement, different relative phase (a vs a').
   Anatase (101): four O2 planes per repeat unit.  The O2 on top of the
   correct termination has Ti 0.73 A below it; another phase of the same O2
   plane has Ti only 0.15 A below.  selection="relative" (default) keeps the
   termination, selection="shape" mixes both.

2. Same relative phase, different absolute phase (a---a vs a---a').
   Rutile IrO2 (110): one repeat unit up is half a cell sideways (the repeat
   vector is tilted), so the top bridging-O row lies exactly over the bottom
   one for 1, 3, 5 layers and half a cell off for 2, 4, 6.  It is the same
   crystal plane either way (same relative phase); selection="absolute" keeps
   only the thicknesses in phase with the input slab.

3. A rotation is a phase too.
   Rutile IrO2 (001): the two IrO2 planes of a repeat unit are one
   arrangement rotated by 90 degrees, IrO2 and IrO2'.

Writes structures to example/output_phases/.  Pass --view to open them in the
ASE GUI (top view, along the surface normal: you see the in-plane registry;
for a side view run e.g. ``ase gui -R -90x example/output_phases/<file>``).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.io import read, write

from _requires import require_taskerslabgen

require_taskerslabgen()  # clear message if Python imports an older taskerslabgen

from taskerslabgen import cutslab, generate_slabs_for_miller


HERE = Path(__file__).resolve().parent
BULKS = HERE / ".." / "bulk_files"
OUT = HERE / "output_phases"


def ti_depth(slab):
    """Depth (A) of the topmost Ti below the topmost atom."""
    z = slab.positions[:, 2]
    return float(z.max() - z[slab.numbers == 22].max())


def save(name, slabs, repeat=(3, 3, 1)):
    """Write *slabs* repeated in-plane, so the registry is easy to see."""
    path = OUT / f"{name}.traj"
    write(path.as_posix(), [s * repeat for s in slabs])
    return path


def case1():
    print("\n1. Anatase (101): same arrangement, different relative phase")
    bulk = read((BULKS / "TiO2_anatase.cif").as_posix())
    q = {"Ti": 4.0, "O": -2.0}
    # A loose dipole tolerance (as for relaxed slabs) lets polar-ish cuts
    # through, so only the plane phase keeps the termination.
    term = generate_slabs_for_miller(bulk, q, (1, 0, 1), [4], dipole_tol=0.3)[(1, 0, 1)][0]
    thick = term["atoms"][0]
    print(f"   thick slab: bottom {term['plane_type']}, top {term['top_plane_type']}")
    views = {}
    for selection in ("relative", "shape"):
        subs = cutslab(thick, q, dipole_tol=0.3, selection=selection)
        print(f"   selection={selection!r}:")
        for s in subs:
            print(f"     {len(s):3d} atoms  top {s.info['cut_top_plane']:6s} "
                  f"Ti {ti_depth(s):.2f} A below the top O")
        views[selection] = subs
    # One structure per file: the first slab with the wrong termination, and
    # the correct slab closest to it in thickness.
    wrong = next(s for s in views["shape"] if abs(ti_depth(s) - ti_depth(thick)) > 0.1)
    right = min(views["relative"], key=lambda s: (abs(len(s) - len(wrong)), -len(s)))
    print(f"   saved: right termination {len(right)} atoms, wrong termination {len(wrong)} atoms")
    return [save("anatase101_right_termination", [right]),
            save("anatase101_wrong_termination", [wrong])]


def case2():
    print("\n2. IrO2 (110): same relative phase, different absolute phase")
    bulk = read((BULKS / "IrO2_rutile.cif").as_posix())
    q = {"Ir": 4.0, "O": -2.0}
    term = generate_slabs_for_miller(bulk, q, (1, 1, 0), [4])[(1, 1, 0)][0]
    thick = term["atoms"][0]
    print(f"   thick slab (4 layers): bottom {term['plane_type']}, top {term['top_plane_type']}")
    for selection in ("relative", "absolute"):
        subs = cutslab(thick, q, selection=selection)
        print(f"   selection={selection!r}:")
        for s in subs:
            over = s.info["cut_phase_overlap"]
            kind = "top exactly over bottom (a---a)" if over > 0.9 else "top half a cell off (a---a')"
            print(f"     {len(s):3d} atoms  overlap top/bottom {over:.2f}  {kind}")
        if selection == "relative":
            one, two = subs[0], subs[1]
    return [save("IrO2_110_1layer_in_phase", [one]), save("IrO2_110_2layers_out_of_phase", [two])]


def case3():
    print("\n3. IrO2 (001): a rotation is a phase")
    bulk = read((BULKS / "IrO2_rutile.cif").as_posix())
    q = {"Ir": 4.0, "O": -2.0}
    term = generate_slabs_for_miller(bulk, q, (0, 0, 1), [3])[(0, 0, 1)][0]
    thick = term["atoms"][0]
    print(f"   planes of one repeat unit: {thick.info['stacking_labels']}")
    for selection in ("relative", "shape"):
        subs = cutslab(thick, q, selection=selection)
        print(f"   selection={selection!r}: "
              + ", ".join(f"{len(s)} atoms ({s.info['cut_bottom_plane']}/{s.info['cut_top_plane']})" for s in subs))
    return [save("IrO2_001_rotated_phases", [thick])]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--view", action="store_true", help="Open the structures in the ASE GUI.")
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    files = case1() + case2() + case3()
    print(f"\nStructures in {OUT}:")
    for f in files:
        print(f"   {f.name}")
    print("Side view: ase gui -R -90x <file>;  top view (in-plane registry): ase gui <file>")

    if args.view:
        from ase.visualize import view

        for f in files:
            view(read(f.as_posix(), index=":"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
