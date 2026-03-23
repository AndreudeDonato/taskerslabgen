"""
Tandem genslab + cutslab workflow with Tasker III reconstruction.

1. generate_slabs_for_miller produces a thick (10-layer) Tasker III slab
   from CeO2x2.cif along (0,0,1), including reconstruction metadata.
2. cutslab receives the *genslab result dict* as reference_termination,
   names planes consistently with genslab, and applies the same
   reconstruction pattern to every newly exposed interior surface so
   that all thinner sub-slabs share identical terminations.
"""
from pathlib import Path

from ase.io import read

from taskerslabgen import generate_slabs_for_miller, cutslab


def main():
    bulk_path = Path("../bulk_files/CeO2x2.cif")
    charges = {"Ce": 4.0, "O": -2.0}
    miller = (0, 0, 1)

    output_dir = Path("output_refcut")
    output_dir.mkdir(parents=True, exist_ok=True)

    bulk = read(bulk_path.as_posix())

    # -- Step 1: generate a thick reference slab (Tasker III) --
    print("=" * 60)
    print("Step 1: generating thick reference slab with genslab")
    print("=" * 60)
    genslab_result = generate_slabs_for_miller(
        bulk_atoms=bulk,
        charges=charges,
        miller=miller,
        layer_thickness_list=[10],
        bulk_name=bulk_path.stem,
        repeat=(1, 1, 1),
        vacuum=15.0,
        out_dir=output_dir.as_posix(),
        plot_out_dir=output_dir.as_posix(),
        layers=1,
        verbose=False,
        output_ext="in",
        prefer_plane=("O",),  # CeO2 001: select plane with oxygen content
    )

    thick_slab = genslab_result["slab_atoms"][0]
    print(f"\nThick slab: {len(thick_slab)} atoms, "
          f"tasker type: {genslab_result['tasker_type']}")
    if genslab_result.get("reconstruction"):
        recon = genslab_result["reconstruction"]
        print(f"  reconstruction plane: {recon['cut_plane_name']}")
        print(f"  delete_info: {len(recon['delete_info'])} atoms per surface")
        print(f"  plane names: {recon['plane_names']}")

    # -- Step 2: cutslab with genslab result as reference --
    print("\n" + "=" * 60)
    print("Step 2: cutslab with reference termination (genslab dict)")
    print("=" * 60)
    cut_result = cutslab(
        input_structure=thick_slab,
        charges=charges,
        axis=2,
        dipole_tol=1e-1,
        save_files=True,
        output_ext="in",
        out_dir=output_dir.as_posix(),
        plot_out_dir=output_dir.as_posix(),
        plot=True,
        verbose=True,
        reference_termination=genslab_result,
        plane_tol=0.3,
    )

    print(f"\nGenerated {len(cut_result['slab_atoms'])} sub-slabs "
          f"(type: {cut_result['tasker_type']})")
    for i, slab in enumerate(cut_result["slab_atoms"]):
        print(f"  slab {i + 1}: {len(slab)} atoms")


if __name__ == "__main__":
    main()
