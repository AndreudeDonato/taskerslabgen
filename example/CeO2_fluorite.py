"""
Generate all dipole-zero Tasker III reconstructions for CeO2 (001)
using generate_slabs_for_miller with prefer_plane="all" and plot=True.
"""
from pathlib import Path

from ase.io import read
from ase.visualize import view

from taskerslabgen import generate_slabs_for_miller


def main():
    bulk_path = Path("../bulk_files/CeO2_fluorite.cif")
    charges = {"Ce": 4.0, "O": -2.0}
    miller = (0, 0, 1)

    output_dir = Path("output_tasker3")
    output_dir.mkdir(parents=True, exist_ok=True)

    bulk = read(bulk_path.as_posix())

    result = generate_slabs_for_miller(
        bulk,
        charges,
        millers=miller,
        layer_thickness_list=[2],
        bulk_name="CeO2",
        plane_tol=0.1,
        vacuum=15.0,
        plot=True,
        plot_out_dir=output_dir.as_posix(),
        verbose=True,
        bond_distances={"Ce-Ce": None, "O-O": None, "Ce-O": 2.35},
        prefer_plane="all",
    )

    slabs = []
    for miller_key, terminations in result.items():
        print(f"\nMiller {miller_key}: {len(terminations)} termination(s)")
        for tid, info in terminations.items():
            slab = info["atoms"][0]
            slabs.append(slab)
            print(
                f"  ID {tid}: type={info['tasker_type']}  "
                f"plane={info['plane_type']}  atoms={len(slab)}"
            )

    if slabs:
        view(slabs)
    else:
        print("No slabs generated.")


if __name__ == "__main__":
    main()
