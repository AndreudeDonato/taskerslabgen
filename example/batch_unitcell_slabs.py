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

from taskerslabgen import cutslab, generate_slabs_for_miller
from taskerslabgen.advanced import compute_reduced_counts, is_stoichiometric_sequence

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
    "albite": [(0, 0, 1), (0, 1, 0), (1, 0, 0)],
}

# Used for bulks whose crystal type is not listed above
DEFAULT_MILLERS = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]

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
    "NaAlSi3O8_albite": "albite",
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
    "Na": 1.0,
    "Al": 3.0,
    "Si": 4.0,
    "O": -2.0,
}

# Per-stem bond filters (None = never count that pair as a bond)
BOND_DISTANCES_BY_STEM = {
    "CeO2_fluorite": {"Ce-Ce": None, "O-O": None, "Ce-O": 2.35},
}
DEFAULT_BOND_DISTANCES = {"O-O": None}

# Per (stem, miller): prefer_plane for generate_slabs_for_miller
PREFER_PLANE = {
    ("CeO2_fluorite", (0, 0, 1)): "O",
}

# Per (stem, miller): plane_tol (angstrom) for both genslab and cutslab.
# The default (None = 0.1 A) merges atoms closer than 0.1 A along the normal
# into one plane, which absorbs relaxation noise.  Marcasite (001) has two O
# planes only 0.07 A apart: merged, the facet needs a Tasker III
# reconstruction; split, it has a better Tasker II cut between them.
# MoO2 forms Mo-Mo dimers, so a rutile-cell MoO2 has no clean planes for any
# tolerance; some facets need the DIPOLE_TOL_MAX fallback below.
PLANE_TOL = {
    ("PtO2_marcasite", (0, 0, 1)): 0.05,
}

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
WORKBULKFILES = REPO_ROOT / "workbulkfiles" / "unitcell"
SHIPPED_BULKS = REPO_ROOT / "bulk_files"
OUTPUT_DIR = REPO_ROOT / "X1output_slabs"
THICK_LAYERS = 6
VACUUM = 15.0
OUTPUT_EXT = "in"
# dipole_tol is |dipole| per formula unit (e*A); polar repeat units are ~1-6.
# Ideal CIF bulks need only the default 0.05; DFT-relaxed bulks may carry
# small symmetry-breaking noise, hence the looser value here.
DIPOLE_TOL_GENSLAB = 0.3
DIPOLE_TOL_CUTSLAB = 0.3
# Facets with no slab within DIPOLE_TOL_GENSLAB are rebuilt with the smallest
# tolerance that works, up to this cap, with a warning (e.g. dimerised MoO2
# needs ~0.45).  Genuinely polar facets need ~1-6, so they still fail.
DIPOLE_TOL_MAX = 1.0
# How cutslab keeps the termination of the thick slab: "relative" cuts only at
# copies of its surface planes (same arrangement and phase in the crystal);
# "shape" also at the same arrangement shifted or rotated, which adds the
# slabs ending half a repeat unit off (0.4 behaviour, mixed stackings).
SELECTION = "relative"
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


def get_crystal_type(stem: str) -> str | None:
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
    if "albite" in stem:
        return "albite"
    return None


def process_miller(
    bulk,
    stem: str,
    miller: tuple[int, int, int],
    thick: int,
    output_dir: Path,
    reduced_counts: dict,
    plot: bool = False,
) -> tuple[int, str | None]:
    """
    Run genslab + cutslab for one bulk/Miller pair.

    Returns (n_slabs_written, error_message).
    """
    h, k, l = miller
    hkl_str = "".join(str(i) for i in miller)
    prefer_plane = PREFER_PLANE.get((stem, miller))
    # Same plane_tol in genslab and cutslab, so both see the same planes.
    plane_tol = PLANE_TOL.get((stem, miller))
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
        plot=plot,
        plot_out_dir=output_dir.as_posix(),
        verbose=VERBOSE,
        bond_distances=bond_distances,
        plane_tol=plane_tol,
        dipole_tol=DIPOLE_TOL_GENSLAB,
        dipole_tol_max=DIPOLE_TOL_MAX,
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
    # Cut with the tolerance the slab was built with (larger if genslab
    # needed the DIPOLE_TOL_MAX fallback).
    dipole_tol_cut = max(DIPOLE_TOL_CUTSLAB, term["dipole_tol"])
    if term["dipole_tol"] > DIPOLE_TOL_GENSLAB:
        print(f"  Slightly polar facet: built with dipole_tol={term['dipole_tol']}")

    print(f"\n  Cutting thick slab for {miller}...")
    sub_slabs = cutslab(
        input_structure=thick_slab,
        charges=CHARGES,
        axis=2,
        dipole_tol=dipole_tol_cut,
        plot=plot,
        plot_out_dir=output_dir.as_posix(),
        cut_at="termination",
        reconstruction=term.get("reconstruction"),
        plane_tol=plane_tol,
        vacuum=VACUUM,
        cuts="right",
        selection=SELECTION,
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


def _discover_bulk_files(
    quick: bool = False, bulk_dir: Path | None = None
) -> tuple[list[Path], Path]:
    """
    Use ``*.out`` / ``*.cif`` files in *bulk_dir* when given.  Otherwise
    prefer FHI-aims ``.out`` files in workbulkfiles/unitcell and fall back
    to the shipped CIF files in bulk_files/.
    """
    if bulk_dir is not None:
        files = sorted(bulk_dir.glob("*.out")) + sorted(bulk_dir.glob("*.cif"))
        if quick:
            files = [p for p in files if p.stem == "CeO2_fluorite"][:1]
        return files, bulk_dir

    out_files = sorted(WORKBULKFILES.glob("*.out"))
    if quick:
        preferred = WORKBULKFILES / "CeO2_fluorite.out"
        if preferred.exists():
            return [preferred], WORKBULKFILES
        cif = SHIPPED_BULKS / "CeO2_fluorite.cif"
        if cif.exists():
            return [cif], SHIPPED_BULKS
        return [], WORKBULKFILES

    if out_files:
        return out_files, WORKBULKFILES

    cif_files = sorted(SHIPPED_BULKS.glob("*.cif"))
    # Prefer primitive unit-cell examples over supercells for batch demos.
    cif_files = [p for p in cif_files if "supercell" not in p.stem.lower()]
    return cif_files, SHIPPED_BULKS


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Batch slab generation. Uses workbulkfiles/unitcell/*.out when "
            "present, otherwise shipped bulk_files/*.cif."
        )
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test: only CeO2_fluorite, Miller (1,1,1)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Write stacking-axis PNG plots next to slab outputs.",
    )
    parser.add_argument(
        "--bulk-dir",
        type=Path,
        default=None,
        help="Read *.out / *.cif bulks from this directory instead.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Where to write the slabs (default: {OUTPUT_DIR}).",
    )
    args = parser.parse_args()
    plot = bool(args.plot)
    output_dir = args.out_dir

    bulk_files, bulk_dir = _discover_bulk_files(
        quick=args.quick, bulk_dir=args.bulk_dir
    )
    if args.quick:
        miller_override = {"CeO2_fluorite": [(1, 1, 1)]}
        thick = 5
        print("Quick mode: CeO2_fluorite, (1,1,1), thick=5")
    else:
        miller_override = None
        thick = THICK_LAYERS

    output_dir.mkdir(parents=True, exist_ok=True)

    if not bulk_files:
        print("No bulk inputs found.")
        if args.bulk_dir is not None:
            print(f"  Looked for *.out / *.cif in {args.bulk_dir}")
        else:
            print(f"  Looked for *.out in {WORKBULKFILES}")
            print(f"  Looked for *.cif in {SHIPPED_BULKS}")
        print("See example/BATCH_SLABS.md for setup instructions.")
        return 1

    print(f"Processing {len(bulk_files)} bulk file(s) from {bulk_dir}")
    print(f"Workflow: generate_slabs_for_miller (thick={thick}) -> cutslab")
    print(f"Output directory: {output_dir}")
    print()

    total_slabs = 0
    errors = []

    for bulk_path in bulk_files:
        stem = bulk_path.stem
        if miller_override and stem in miller_override:
            millers = miller_override[stem]
        else:
            crystal = get_crystal_type(stem)
            if crystal is None:
                print(f"{stem}: crystal type not listed, using DEFAULT_MILLERS")
                millers = DEFAULT_MILLERS
            else:
                millers = MILLER_BY_CRYSTAL[crystal]

        try:
            bulk = read(bulk_path.as_posix())
            _ensure_charges_for_atoms(CHARGES, bulk)
        except Exception as exc:
            errors.append((stem, str(exc)))
            continue

        reduced_counts = _bulk_reduced_counts(bulk, CHARGES)

        for miller in millers:
            try:
                n_written, err = process_miller(
                    bulk,
                    stem,
                    miller,
                    thick,
                    output_dir,
                    reduced_counts,
                    plot=plot,
                )
                total_slabs += n_written
                if err:
                    errors.append((f"{stem} {miller}", err))
            except Exception as exc:
                errors.append((f"{stem} {miller}", str(exc)))

    print(f"Generated {total_slabs} slab files in {output_dir}")
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for item, msg in errors:
            print(f"  {item}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
