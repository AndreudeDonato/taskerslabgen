"""
Generate one Tasker I/II surface per Miller index for IrO2 and visualize.
"""
from pathlib import Path

from ase.io import read
from ase.visualize import view

from taskerslabgen import generate_slabs_for_miller


def main():
    bulk_path = Path("../bulk_files/IrO2_rutile.cif")
    charges = {"Ir": 4.0, "O": -2.0}

    millers = [
        (1, 1, 1),
        (0, 0, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
    ]

    output_dir = Path("output_tasker2")
    output_dir.mkdir(parents=True, exist_ok=True)

    bulk = read(bulk_path.as_posix())

    result = generate_slabs_for_miller(
        bulk_atoms=bulk,
        charges=charges,
        millers=millers,
        layer_thickness_list=[5],
        bulk_name=bulk_path.stem,
        vacuum=15.0,
        plot_out_dir=output_dir.as_posix(),
        verbose=True,
        plot=True,
    )

    slabs = []
    for miller in millers:
        terminations = result[miller]
        if not terminations:
            print(f"  No termination found for {miller}")
            continue
        first_tid = min(terminations.keys())
        slab = terminations[first_tid]["atoms"][0]
        ttype = terminations[first_tid]["tasker_type"]
        print(f"  {miller}: {len(slab)} atoms, Tasker {ttype}")
        slabs.append(slab)

    if slabs:
        view(slabs)


if __name__ == "__main__":
    main()
