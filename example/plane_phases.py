"""
Plane phases: what the labels O, O', O'' mean and how selection= uses them.

Every plane is treated as a wave: a smooth periodic density with one Gaussian
per atom.  Two planes have the same *arrangement* when one wave overlaps the
other after some shift or rotation; they are in the same *phase* when they
overlap as they are.  A label is the arrangement plus its phase:

    O, O', O''      one arrangement of O atoms in three phases (shifted or rotated)
    IrO2-a, IrO2-b  two different arrangements of IrO2 (no shift or rotation maps them)

Three cases, each cut with two selections:

1. Same arrangement, different relative phase (a vs a').
   Anatase (101): four O2 planes per repeat unit.  Over the correct
   termination the top O sits 0.73 A above the Ti; over another phase of the
   same O2 plane only 0.15 A.  "relative" (default) keeps the termination,
   "shape" mixes both.  The mixed slabs, from O2 Ti2 O2' (6 atoms) up, have
   a polarity of 0.022 /A at every thickness, which passes dipole_tol=0.05
   (the value relaxed slabs need), so only the phase keeps the termination.

2. Same relative phase, different absolute phase (a---a vs a---a').
   Rutile IrO2 (110): one repeat unit up is half a cell sideways (the repeat
   vector is tilted), so the top bridging-O row lies exactly over the bottom
   one for 1, 3 layers and half a cell off for 2, 4.  It is the same crystal
   plane either way; "absolute" keeps only the thicknesses in phase with the
   input slab.

3. A rotation is a phase too.
   Rutile IrO2 (001): the two IrO2 planes of a repeat unit are one
   arrangement rotated by 90 degrees, IrO2 and IrO2'.

Output in example/output_phases/:

- plane_phases.traj: every sub-slab of the three cases, one frame each,
  repeated 3x3 in-plane so the registry is easy to see (the frame table is
  printed; each frame's description is also in atoms.info).
- one folder per case with a plot of where the bulk was cut for the thick
  slab, and per selection a plot of every cut (planes allowed as bottom
  surface in red, as top surface in blue).

Pass --view to open the trajectory in the ASE GUI (top view along the
normal; for a side view run ``ase gui -R -90x example/output_phases/plane_phases.traj``).
"""
from __future__ import annotations

import argparse
from pathlib import Path

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


def run_case(folder, title, bulk_file, charges, miller, layers, selections,
             describe, gen_kwargs=None, cut_kwargs=None):
    """Build the thick slab, cut it with each selection (pictures in
    OUT/folder/<selection>/) and return the frames: the whole series of
    every selection, one after the other."""
    print(f"\n{title}")
    bulk = read((BULKS / bulk_file).as_posix())
    term = generate_slabs_for_miller(bulk, charges, miller, [layers], bulk_name=folder,
                                     plot=True, plot_out_dir=(OUT / folder).as_posix(),
                                     **(gen_kwargs or {}))[miller][0]
    thick = term["atoms"][0]
    print(f"   thick slab: bottom {term['plane_type']}, top {term['top_plane_type']}, "
          f"planes of one repeat unit {thick.info['stacking_labels']}")
    case = title.split(":")[0]
    frames = []
    for selection in selections:
        subs = cutslab(thick, charges, selection=selection, plot=True,
                       plot_out_dir=(OUT / folder / selection).as_posix(), **(cut_kwargs or {}))
        print(f"   selection={selection!r}:")
        for s in subs:
            text = describe(s)
            print(f"     {len(s):3d} atoms  {s.info['cut_bottom_plane']} ... "
                  f"{s.info['cut_top_plane']:7s} {text}")
            frames.append((f"{case} selection={selection!r}: {len(s)} atoms, "
                           f"{s.info['cut_bottom_plane']} ... {s.info['cut_top_plane']}, {text}", s))
    return frames


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--view", action="store_true", help="Open the trajectory in the ASE GUI.")
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    q_ti, q_ir = {"Ti": 4.0, "O": -2.0}, {"Ir": 4.0, "O": -2.0}
    frames = []
    # The dipole tolerance relaxed slabs need (0.05) lets the slightly
    # polar mixed cuts through, so only the plane phase keeps the termination.
    frames += run_case(
        "1_anatase101", "1. Anatase (101): same arrangement, different relative phase",
        "TiO2_anatase.cif", q_ti, (1, 0, 1), 3, ("relative", "shape"),
        lambda s: f"Ti {ti_depth(s):.2f} A below the top O",
        gen_kwargs={"dipole_tol": 0.05}, cut_kwargs={"dipole_tol": 0.05},
    )
    frames += run_case(
        "2_IrO2_110", "2. IrO2 (110): same relative phase, different absolute phase",
        "IrO2_rutile.cif", q_ir, (1, 1, 0), 4, ("relative", "absolute"),
        lambda s: ("top exactly over bottom (a---a)" if s.info["cut_phase_overlap"] > 0.9
                   else "top half a cell off (a---a')"),
    )
    frames += run_case(
        "3_IrO2_001", "3. IrO2 (001): a rotation is a phase",
        "IrO2_rutile.cif", q_ir, (0, 0, 1), 3, ("relative", "shape"),
        lambda s: ("top exactly over bottom" if s.info["cut_phase_overlap"] > 0.9
                   else "top rotated from bottom"),
    )

    images = []
    for text, slab in frames:
        image = slab * (3, 3, 1)
        image.info = {"description": text}
        images.append(image)
    traj = OUT / "plane_phases.traj"
    write(traj.as_posix(), images)

    print(f"\n{traj}  (frames repeated 3x3 in-plane):")
    for k, (text, _) in enumerate(frames):
        print(f"   frame {k:2d}: {text}")
    print(f"Cut plots: {OUT}/<case>/<selection>/*.png")
    print(f"Side view: ase gui -R -90x {traj}")

    if args.view:
        from ase.visualize import view

        view(images)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
