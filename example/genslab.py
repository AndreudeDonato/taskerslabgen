from pathlib import Path

import numpy as np
from ase.io import read

from taskerslabgen import generate_slabs_for_miller, parse_hirshfeld_fhi_aims


def main():
    bulk_path = Path("../bulk_files/MoO2_rutile.out")
    charges_path = Path("../bulk_files/MoO2_rutile.out")
    
    millers = [
        (1, 1, 1),
        (0, 0, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
    ]

    plane_tol = 0.1
    charge_tol = 1e-3
    dipole_tol = 1e-6
    layer_thickness_list = [13]
    vacuum = 15.0
    out_dir = Path(".")
    output_dir = out_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)


    bulk = read(bulk_path.as_posix())
    charges = parse_hirshfeld_fhi_aims(charges_path.as_posix())
    bulk_stem = bulk_path.stem
    for miller in millers:
        generate_slabs_for_miller(
            bulk_atoms=bulk,
            charges=charges,
            miller=miller,
            layer_thickness_list=layer_thickness_list,
            bulk_name=bulk_stem,
            repeat=(1, 1, 1),
            vacuum=vacuum,
            out_dir=output_dir.as_posix(),
            plot_out_dir=output_dir.as_posix(),
            layers=1,
            plane_tol=plane_tol,
            charge_tol=charge_tol,
            dipole_tol=dipole_tol,
            verbose=True,
            output_ext="in"
        )


if __name__ == "__main__":
    main()
