from pathlib import Path
import sys
from ase.io import read
from taskerslabgen import cutslab, parse_hirshfeld_fhi_aims


def main():

    out_dir = Path(".")
    output_dir = out_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_file = sys.argv[1]
    charges = {
        "Ce": 4.0,
        "Ir": 4.0,
        "Os": 4.0,
        "Pb": 4.0,
        "Pd": 4.0,
        "Pt": 4.0,
        "Ru": 4.0,
        "Sn": 4.0,
        "Ti": 4.0,
        "V": 4.0,
        "Mo": 4.0,
        "O": -2.0,
    }
    result = cutslab(
        input_structure=input_file,
        charges=charges,
        axis=2,
        dipole_tol=9e-1,
        save_files=True,
        output_ext="in",
        out_dir=output_dir,
        plot_out_dir=output_dir,
        plot=True,
        verbose=True,
    
    )
    

    print(f"Generated {len(result['slab_atoms'])} cut slabs.")


if __name__ == "__main__":
    main()
