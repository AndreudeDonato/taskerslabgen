from pathlib import Path

from ase.io import read
from ase.visualize import view

from taskerslabgen import generate_slabs_for_miller, parse_hirshfeld_fhi_aims


def main():
    bulk_path = Path("../bulk_files/CeO2.cif")

    bulk = read(bulk_path.as_posix())
    charges = {"Ce": 4, "O": -2}

    result = generate_slabs_for_miller(
        bulk_atoms=bulk,
        charges=charges,
        miller=(0, 0, 1),
        plane_tol = 0.1,
        charge_tol = 1e-3,
        dipole_tol = 1e-6,
        layer_thickness_list=[2],
        bulk_name=bulk_path.stem,
        repeat=(1, 1, 1),
        bond_distances={'Ce-Ce': None,'O-O': None, 'Ce-O': 2.35},
        output_ext=None,
        verbose=True,
        plot=True,
    )

    slab = result["slab_atoms"][0]
    view(slab)


if __name__ == "__main__":
    main()
