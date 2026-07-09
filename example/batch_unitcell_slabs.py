"""
Batch genslab + cutslab workflow for multiple unit-cell bulk structures.

For each bulk file and Miller index:
  1. generate_slabs_for_miller builds a thick reference slab (best termination).
  2. cutslab cuts it into thinner sub-slabs preserving the same termination
     (including Tasker III reconstruction when applicable).
  3. Each sub-slab is saved as:

     {stem}_hkl_{h}{k}{l}_cut_{stoich_k}.in

Usage (after ``pip install -e .``):
  python example/batch_unitcell_slabs.py
  python example/batch_unitcell_slabs.py --quick
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from ase.data import atomic_numbers
from ase.io import read, write

from taskerslabgen import (
    compute_reduced_counts,
    cutslab,
    generate_slabs_for_miller,
    is_stoichiometric_sequence,
)

# -----------------------------------------------------------------------------
# Miller indices per crystal type
# -----------------------------------------------------------------------------
MILLER_BY_CRYSTAL = {
    "rutile": [(1, 1, 1), (0, 0, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0)],
    "CeO2_fluorite": [(0, 0, 1), (1, 1, 1), (1, 1, 0)],
    "PtO2_marcasite": [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
    ],
    "TiO2_anatase": [(1, 0, 1), (0, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 2)],
    "PbO2_brookite": [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, 0, 1),
        (2, 1, 0),
    ],
    "VO2_C2m": [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (0, 1, 1),
        (1, 0, 1),
    ],
    "VO2_P21c": [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (0, 1, 1),
        (1, 0, 1),
    ],
    "OsO2_pyrite": [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 0), (2, 1, 1)],
}

STEM_TO_CRYSTAL = {
    "CeO2_fluorite": "CeO2_fluorite",
    "IrO2_rutile": "rutile",
    "MoO2_rutile": "rutile",
    "OsO2_pyrite": "OsO2_pyrite",
    "OsO2_rutile": "rutile",
    "PbO2_rutile": "rutile",
    "PdO2_rutile": "rutile",
    "PtO2_rutile": "rutile",
    "PtO2_marcasite": "PtO2_marcasite",
    "RuO2_rutile": "rutile",
    "SnO2_rutile": "rutile",
    "TiO2_anatase": "TiO2_anatase",
    "TiO2_rutile": "rutile",
    "VO2_C2m": "VO2_C2m",
    "VO2_P2c": "VO2_P21c",
    "VO2_rutile": "rutile",
}

CHARGES = {
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

# Per-stem bond filters (None = forbid that pair in Tasker III adjacency)
BOND_DISTANCES_BY_STEM = {
    "CeO2_fluorite": {"Ce-Ce": None, "O-O": None, "Ce-O": 2.35},
}
DEFAULT_BOND_DISTANCES = {"O-O": None}

# Per (stem, miller): prefer_plane for generate_slabs_for_miller
PREFER_PLANE = {
    ("CeO2_fluorite", (0, 0, 1)): "O",
}

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
WORKBULKFILES = REPO_ROOT / "workbulkfiles" / "unitcell"
OUTPUT_DIR = REPO_ROOT / "X1output_slabs"
PLANE_TOL_GENSLAB = 0.005
PLANE_TOL_CUTSLAB = 0.05
THICK_LAYERS = 6
VACUUM = 15.0
OUTPUT_EXT = "in"
DIPOLE_TOL_GENSLAB = 9e-1
DIPOLE_TOL_CUTSLAB = 1e-1
VERBOSE = False


def _ensure_charges_for_atoms(charges_dict, atoms):
    symbols = set(atoms.get_chemical_symbols())
    missing = symbols - set(charges_dict.keys())
    if missing:
        raise ValueError(
            f"Charges dict missing entries for: {missing}. "
            "Add them to CHARGES in batch_unitcell_slabs.py"
        )


def _bulk_reduced_counts(bulk_atoms, charges):
    charge_by_Z = {
        atomic_numbers[sym]: float(charges[sym])
        for sym in charges
        if sym in atomic_numbers
    }
    atoms_z = np.array(
        [
            [num, 0.0, charge_by_Z.get(int(num), 0.0)]
            for num in bulk_atoms.numbers
        ]
    )
    return compute_reduced_counts(atoms_z)


def _stoich_k_for_slab(slab, reduced_counts):
    counts = Counter(int(z) for z in slab.numbers)
    is_stoich, k = is_stoichiometric_sequence(dict(counts), reduced_counts)
    if is_stoich and k is not None:
        return k
    return slab.info.get("cut_n_planes", 0)


def get_crystal_type(stem: str) -> str:
    if stem in STEM_TO_CRYSTAL:
        return STEM_TO_CRYSTAL[stem]
    if "fluorite" in stem:
        return "CeO2_fluorite"
    if "marcasite" in stem:
        return "PtO2_marcasite"
    if "anatase" in stem:
        return "TiO2_anatase"
    if "brookite" in stem:
        return "PbO2_brookite"
    if "C2m" in stem:
        return "VO2_C2m"
    if "P2c" in stem:
        return "VO2_P21c"
    if "pyrite" in stem:
        return "OsO2_pyrite"
    if "rutile" in stem:
        return "rutile"
    raise ValueError(f"Unknown crystal type for stem: {stem}")


def process_miller(
    bulk,
    stem: str,
    miller: tuple[int, int, int],
    thick: int,
    output_dir: Path,
    reduced_counts: dict,
) -> tuple[int, str | None]:
    """
    Run genslab + cutslab for one bulk/Miller pair.

    Returns (n_slabs_written, error_message).
    """
    h, k, l = miller
    hkl_str = "".join(str(i) for i in miller)
    prefer_plane = PREFER_PLANE.get((stem, miller))
    bond_distances = BOND_DISTANCES_BY_STEM.get(stem, DEFAULT_BOND_DISTANCES)

    print("=" * 60)
    print(f"{stem}  Miller {miller}")
    print("=" * 60)

    genslab_result = generate_slabs_for_miller(
        bulk_atoms=bulk,
        charges=CHARGES,
        millers=miller,
        layer_thickness_list=[thick],
        bulk_name=stem,
        vacuum=VACUUM,
        plot=True,
        plot_out_dir=output_dir.as_posix(),
        verbose=VERBOSE,
        bond_distances=bond_distances,
        plane_tol=PLANE_TOL_GENSLAB,
        dipole_tol=DIPOLE_TOL_GENSLAB,
        prefer_plane=prefer_plane,
        candidates="best",
    )

    terminations = genslab_result[miller]
    if not terminations:
        return 0, "No termination found"

    tid = min(terminations.keys())
    term = terminations[tid]
    thick_slab = term["atoms"][0]
    print(
        f"\n  Thick slab: {len(thick_slab)} atoms, "
        f"Tasker {term['tasker_type']}, plane={term['plane_type']}"
    )

    print(f"\n  Cutting thick slab for {miller}...")
    sub_slabs = cutslab(
        input_structure=thick_slab,
        charges=CHARGES,
        axis=2,
        dipole_tol=DIPOLE_TOL_CUTSLAB,
        plot=True,
        plot_out_dir=output_dir.as_posix(),
        cut_at="termination",
        reconstruction=term.get("reconstruction"),
        plane_tol=PLANE_TOL_CUTSLAB,
        vacuum=VACUUM,
        cuts="right",
        verbose=VERBOSE,
    )

    ext = OUTPUT_EXT.lstrip(".")
    n_written = 0
    print(f"\n  Generated {len(sub_slabs)} sub-slabs for {miller}")
    for slab in sub_slabs:
        stoich_k = _stoich_k_for_slab(slab, reduced_counts)
        fname = f"{stem}_hkl_{hkl_str}_cut_{stoich_k}.{ext}"
        out_path = output_dir / fname
        write(out_path.as_posix(), slab)
        n_written += 1
        if VERBOSE:
            bp = slab.info.get("cut_bottom_plane", "?")
            tp = slab.info.get("cut_top_plane", "?")
            print(f"    saved: {fname}  ({len(slab)} atoms, {bp}-{tp})")

    print()
    return n_written, None


def main():
    parser = argparse.ArgumentParser(
        description="Batch slab generation for workbulkfiles/unitcell"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test: only CeO2_fluorite, Miller (1,1,1)",
    )
    args = parser.parse_args()

    if args.quick:
        bulk_files = [WORKBULKFILES / "CeO2_fluorite.out"]
        miller_override = {"CeO2_fluorite": [(1, 1, 1)]}
        thick = 5
        print("Quick mode: CeO2_fluorite, (1,1,1), thick=5")
    else:
        bulk_files = sorted(WORKBULKFILES.glob("*.out"))
        miller_override = None
        thick = THICK_LAYERS

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not bulk_files or not bulk_files[0].exists():
        print(f"No .out files found in {WORKBULKFILES}")
        print("Place relaxed FHI-aims bulk structures there before running.")
        print("See example/BATCH_SLABS.md for setup instructions.")
        return

    print(f"Processing {len(bulk_files)} bulk file(s) from {WORKBULKFILES}")
    print(f"Workflow: generate_slabs_for_miller (thick={thick}) -> cutslab")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    total_slabs = 0
    errors = []

    for bulk_path in bulk_files:
        stem = bulk_path.stem
        try:
            if miller_override and stem in miller_override:
                millers = miller_override[stem]
            else:
                crystal = get_crystal_type(stem)
                millers = MILLER_BY_CRYSTAL[crystal]
        except (KeyError, ValueError) as exc:
            errors.append((stem, str(exc)))
            continue

        try:
            bulk = read(bulk_path.as_posix())
        except Exception as exc:
            errors.append((stem, f"Read failed: {exc}"))
            continue

        _ensure_charges_for_atoms(CHARGES, bulk)
        reduced_counts = _bulk_reduced_counts(bulk, CHARGES)

        for miller in millers:
            try:
                n_written, err = process_miller(
                    bulk, stem, miller, thick, OUTPUT_DIR, reduced_counts
                )
                total_slabs += n_written
                if err:
                    errors.append((f"{stem} {miller}", err))
            except Exception as exc:
                errors.append((f"{stem} {miller}", str(exc)))

    print(f"Generated {total_slabs} slab files in {OUTPUT_DIR}")
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for item, msg in errors:
            print(f"  {item}: {msg}")


if __name__ == "__main__":
    main()
