"""
Tandem genslab + cutslab workflow with Tasker III reconstruction.

1. generate_slabs_for_miller produces a thick (10-layer) Tasker III slab
   from CeO2.cif along (0,0,1) with O termination.
2. cutslab receives the termination dict as reference_termination and
   applies the same reconstruction pattern to every newly exposed surface,
   returning a list of Atoms objects of decreasing thickness.

For supercells (e.g. CeO2x2.cif), run genslab on the primitive cell first
to avoid the combinatorial explosion in Tasker III candidate enumeration,
then repeat the thick slab before passing to cutslab if needed.
"""
from pathlib import Path

from ase.io import read, write

from taskerslabgen import generate_slabs_for_miller, cutslab


def main():
    bulk_path = Path("../bulk_files/CeO2_fluorite.cif")
    charges = {"Ce": 4.0, "O": -2.0}
    miller = (0, 0, 1)

    output_dir = Path("output_cutslab")
    output_dir.mkdir(parents=True, exist_ok=True)

    bulk = read(bulk_path.as_posix())

    # Step 1: generate a thick reference slab (Tasker III with O termination)
    print("=" * 60)
    print("Step 1: generating thick reference slab with genslab")
    print("=" * 60)
    genslab_result = generate_slabs_for_miller(
        bulk_atoms=bulk,
        charges=charges,
        millers=miller,
        layer_thickness_list=[10],
        bulk_name=bulk_path.stem,
        vacuum=15.0,
        plot_out_dir=output_dir.as_posix(),
        verbose=False,
        prefer_plane="O",
        bond_distances={"Ce-Ce": None, "O-O": None, "Ce-O": 2.35},
    )

    # Extract the termination entry for (0,0,1), plane ID 0
    term = genslab_result[miller][0]
    thick_slab = term["atoms"][0]
    print(f"\nThick slab: {len(thick_slab)} atoms, tasker type: {term['tasker_type']}")
    if term.get("reconstruction"):
        recon = term["reconstruction"]
        print(f"  reconstruction plane: {recon['cut_plane_name']}")
        print(f"  delete_info: {len(recon['delete_info'])} atoms per surface")

    # Step 2: cutslab with reference termination
    print("\n" + "=" * 60)
    print("Step 2: cutslab with reference termination")
    print("=" * 60)
    sub_slabs = cutslab(
        input_structure=thick_slab,
        charges=charges,
        axis=2,
        dipole_tol=1e-1,
        plot_out_dir=output_dir.as_posix(),
        plot=True,
        verbose=True,
        reference_termination=term,
        plane_tol=0.1,
    )

    print(f"\nGenerated {len(sub_slabs)} sub-slabs")
    for i, slab in enumerate(sub_slabs):
        print(f"  slab {i + 1}: {len(slab)} atoms")
        write((output_dir / f"subslab_{i+1}.xyz").as_posix(), slab)


if __name__ == "__main__":
    main()
