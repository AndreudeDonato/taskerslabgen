"""
Tandem genslab + cutslab workflow for CeO2 fluorite supercell.

For each Miller index:
  1. genslab generates a thick reference slab with the best O-terminated plane.
  2. cutslab cuts the thick slab into thinner sub-slabs preserving the
     same termination (including Tasker III reconstruction if applicable).
  3. Each sub-slab is saved with the naming convention:

     {stem}_hkl_{millerindex}_between_{bottom}_{top}_cut_{cutindex}.cif
"""
from pathlib import Path

from ase.io import read, write

from taskerslabgen import generate_slabs_for_miller, cutslab


def main():
    here = Path(__file__).resolve().parent
    bulk_path = here / ".." / "bulk_files" / "CeO2_fluorite_supercell2x2x2.cif"
    charges = {"Ce": 4.0, "O": -2.0}
    millers = [(1, 1, 0), (0, 0, 1)]
    ext = "cif"

    output_dir = here / "output_x2supercell"
    output_dir.mkdir(parents=True, exist_ok=True)

    bulk = read(bulk_path.as_posix())
    stem = bulk_path.stem

    for miller in millers:
        hkl_str = "".join(str(i) for i in miller)
        print("=" * 60)
        print(f"Miller {miller}")
        print("=" * 60)

        # Step 1: generate a thick reference slab (best O-terminated)
        genslab_result = generate_slabs_for_miller(
            bulk_atoms=bulk,
            charges=charges,
            millers=miller,
            layer_thickness_list=[3],
            bulk_name=stem,
            vacuum=15.0,
            plot=True,
            plot_out_dir=output_dir.as_posix(),
            bond_distances={"Ce-Ce": None, "O-O": None, "Ce-O": 2.35},
            prefer_plane="O",
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

        # Step 2: cutslab preserving termination
        #   For Tasker III surfaces, pass reconstruction so that newly
        #   exposed interior planes receive the same atomic deletion.
        print(f"\n  Cutting thick slab for {miller}...")
        sub_slabs = cutslab(
            input_structure=thick_slab,
            charges=charges,
            axis=2,
            dipole_tol=1e-1,
            plot=True,
            plot_out_dir=output_dir.as_posix(),
            cut_at="termination",
            reconstruction=term.get("reconstruction"),
            plane_tol=0.05,
            vacuum=15.0,
            cuts="right",
        )

        # Step 3: save each sub-slab using metadata from cutslab
        print(f"\n  Generated {len(sub_slabs)} sub-slabs for {miller}")
        for i, slab in enumerate(sub_slabs):
            bp = slab.info.get("cut_bottom_plane", "?")
            tp = slab.info.get("cut_top_plane", "?")
            fname = (
                f"{stem}_hkl_{hkl_str}_between_{bp}_{tp}_cut_{i}.{ext}"
            )
            out_path = output_dir / fname
            write(out_path.as_posix(), slab)
            print(f"    saved: {fname}  ({len(slab)} atoms)")

        print()


if __name__ == "__main__":
    main()
