#!/usr/bin/env python3
"""
Batch slab generation script for workbulkfiles.

Workflow (same as cutslab example):
  1. Generate a THICK slab from the relaxed bulk (genslab)
  2. Cut that thick slab down into thinner layers (cutslab with reference)

This ensures all sub-slabs share identical terminations. Metal-metal and
metal-O bonds only (no O-O). Output naming:
  {bulk_name}_hkl_{h}{k}{l}_cut_{stoich_k}.in
  Example: CeO2_fluorite_hkl_111_cut_32.in

Usage:
  python3 custlab.py              # Process all 16 files
  python3 custlab.py --quick      # Quick test: 1 file, 1 Miller
"""
from pathlib import Path
import sys
import argparse

# Ensure taskerslabgen is importable when run from project root
_taskerslabgen_src = Path(__file__).resolve().parent / "taskerslabgen" / "src"
if _taskerslabgen_src.exists() and str(_taskerslabgen_src) not in sys.path:
    sys.path.insert(0, str(_taskerslabgen_src))

from ase.io import read, write

from taskerslabgen import generate_slabs_for_miller, cutslab

# -----------------------------------------------------------------------------
# Miller indices per crystal type
# -----------------------------------------------------------------------------
MILLER_BY_CRYSTAL = {
    "rutile": [(1, 1, 1), (0, 0, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0)],
    "CeO2_fluorite": [(1, 1, 1), (1, 1, 0), (1, 0, 0)],
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

# -----------------------------------------------------------------------------
# Map file stem -> crystal type key for Miller lookup
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Charges (metal +4, O -2)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Bond filter: only metal-metal and metal-O allowed; O-O forbidden
# -----------------------------------------------------------------------------
BOND_DISTANCES = {
    "O-O": None,   # forbid O-O bonds in adjacency matrix
    "Ce-Ce": None,  # forbid Ce-Ce bonds
}

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_TASKERSLABGEN_DIR = _SCRIPT_DIR.parent
WORKBULKFILES = _TASKERSLABGEN_DIR / "workbulkfiles"
OUTPUT_DIR = _TASKERSLABGEN_DIR / "output_slabs"
PLANE_TOL = 0.05
THICK_LAYERS = 3  # thick reference slab (formula units) before cutting
VACUUM = 15.0
OUTPUT_EXT = "in"
DIPOLE_TOL_GENSLAB = 9e-1  # prefer Tasker II over III when |dipole| <= this (e.g. P0-P0 cuts)
DIPOLE_TOL = 9e-1  # relax for cutslab reference-matched cuts (Tasker III)
MATCH_REFERENCE_PLANES = False  # False: cut wherever dipole=0, no plane matching; True: match planes to reference
VERBOSE = False


def _ensure_charges_for_atoms(charges_dict, atoms):
    """Ensure charges dict has entries for all elements in atoms."""
    symbols = set(atoms.get_chemical_symbols())
    missing = symbols - set(charges_dict.keys())
    if missing:
        raise ValueError(
            f"Charges dict missing entries for: {missing}. "
            f"Add them to CHARGES in custlab.py"
        )


def get_crystal_type(stem: str) -> str:
    """Return crystal type key for a given file stem."""
    if stem in STEM_TO_CRYSTAL:
        return STEM_TO_CRYSTAL[stem]
    # Fallback: try to infer from stem
    if "fluorite" in stem:
        return "CeO2_fluorite"
    if "marcasite" in stem:
        return "PtO2_marcasite"
    if "anatase" in stem:
        return "TiO2_anatase"
    if "brookite" in stem or (stem == "PbO2_rutile"):
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


def main():
    parser = argparse.ArgumentParser(description="Batch slab generation for workbulkfiles")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test: only CeO2_fluorite, (1,1,1)",
    )
    args = parser.parse_args()

    if args.quick:
        bulk_files = [WORKBULKFILES / "CeO2_fluorite.out"]
        miller_override = {"CeO2_fluorite": [(1, 1, 1)]}
        thick = 5
        print("Quick mode: CeO2_fluorite, (1,1,1), thick=10")
    else:
        bulk_files = sorted(WORKBULKFILES.glob("*.out"))
        miller_override = None
        thick = THICK_LAYERS

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not bulk_files or not bulk_files[0].exists():
        print(f"No .out files found in {WORKBULKFILES}")
        return

    print(f"Processing {len(bulk_files)} bulk file(s) from {WORKBULKFILES}")
    print(f"Workflow: genslab (thick={thick} layers) → cutslab (all cuts)")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    total_slabs = 0
    errors = []
    n_done = 0

    for bulk_path in bulk_files:
        stem = bulk_path.stem
        try:
            if miller_override and stem in miller_override:
                millers = miller_override[stem]
            else:
                crystal = get_crystal_type(stem)
                millers = MILLER_BY_CRYSTAL[crystal]
        except (KeyError, ValueError) as e:
            errors.append((stem, str(e)))
            continue

        n_done += 1
        print(f"[{n_done}/{len(bulk_files)}] {stem} ({len(millers)} Miller indices)...", flush=True)

        try:
            bulk = read(bulk_path.as_posix())
        except Exception as e:
            errors.append((stem, f"Read failed: {e}"))
            continue

        _ensure_charges_for_atoms(CHARGES, bulk)

        for miller in millers:
            h, k, l = miller
            try:
                # Step 1: generate thick reference slab
                genslab_result = generate_slabs_for_miller(
                    bulk_atoms=bulk,
                    charges=CHARGES,
                    miller=miller,
                    layer_thickness_list=[thick],
                    bulk_name=stem,
                    repeat=(1, 1, 1),
                    out_dir=OUTPUT_DIR.as_posix(),
                    plot_out_dir=OUTPUT_DIR.as_posix(),
                    layers=1,
                    vacuum=VACUUM,
                    plot=True,
                    verbose=VERBOSE,
                    output_ext=None,
                    bond_distances=BOND_DISTANCES,
                    plane_tol=PLANE_TOL,
                    dipole_tol=DIPOLE_TOL_GENSLAB,
                )
                thick_slab = genslab_result["slab_atoms"][0]

                # Step 2: cut thick slab into thinner layers
                cut_result = cutslab(
                    input_structure=thick_slab,
                    charges=CHARGES,
                    axis=2,
                    dipole_tol=DIPOLE_TOL,
                    save_files=False,
                    output_ext=None,
                    out_dir=OUTPUT_DIR.as_posix(),
                    plot_out_dir=OUTPUT_DIR.as_posix(),
                    plot=True,
                    verbose=VERBOSE,
                    reference_termination=genslab_result,
                    plane_tol=PLANE_TOL,
                    cuts="left",
                    match_reference_planes=MATCH_REFERENCE_PLANES,
                )

                # Step 3: write all cuts with crystal structure and hkl in filename
                ext = OUTPUT_EXT.lstrip(".")
                for slab, cut_info in zip(
                    cut_result["slab_atoms"], cut_result["valid_cuts"]
                ):
                    stoich_k = cut_info["stoich_k"]
                    fname = f"{stem}_hkl_{h}{k}{l}_cut_{stoich_k}.{ext}"
                    out_path = OUTPUT_DIR / fname
                    write(out_path.as_posix(), slab)
                    total_slabs += 1

                if VERBOSE:
                    print(f"  {stem} ({h}{k}{l}): {len(cut_result['slab_atoms'])} slabs")
            except Exception as e:
                errors.append((f"{stem} ({h},{k},{l})", str(e)))

    print(f"\nGenerated {total_slabs} slab files in {OUTPUT_DIR}")
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for item, msg in errors:
            print(f"  {item}: {msg}")


if __name__ == "__main__":
    main()
