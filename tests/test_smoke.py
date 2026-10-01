"""
Smoke tests for taskerslabgen.

Lightweight integration tests that verify the main workflows run end-to-end
and produce sensible outputs. Uses the CeO2 / IrO2 / albite bulk files shipped
in the repository.
"""
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

BULK_DIR = Path(__file__).resolve().parent.parent / "bulk_files"
CEO2_PATH = BULK_DIR / "CeO2_fluorite.cif"
IRO2_PATH = BULK_DIR / "IrO2_rutile.cif"
ALBITE_PATH = BULK_DIR / "NaAlSi3O8_albite.cif"
CEO2_CHARGES = {"Ce": 4.0, "O": -2.0}
IRO2_CHARGES = {"Ir": 4.0, "O": -2.0}
ALBITE_CHARGES = {"Na": 1.0, "Al": 3.0, "Si": 4.0, "O": -2.0}
BOND_DISTS = {"Ce-Ce": None, "O-O": None, "Ce-O": 2.35}


@pytest.fixture
def ceo2_bulk():
    return read(CEO2_PATH.as_posix())


@pytest.fixture
def iro2_bulk():
    return read(IRO2_PATH.as_posix())


@pytest.fixture
def albite_bulk():
    return read(ALBITE_PATH.as_posix())


def _formal_charge_sum(atoms, charges):
    return float(sum(charges[sym] for sym in atoms.get_chemical_symbols()))


def _stoich_ratio(atoms, reduced):
    from collections import Counter

    counts = Counter(int(z) for z in atoms.numbers)
    ratios = []
    for Z, red in reduced.items():
        if red == 0:
            continue
        if Z not in counts:
            return None
        ratios.append(counts[Z] / red)
    if not ratios:
        return None
    k = ratios[0]
    if any(abs(r - k) > 1e-8 for r in ratios):
        return None
    return k


# ------------------------------------------------------------------
# 1. Import check
# ------------------------------------------------------------------
def test_import_public_api():
    from taskerslabgen import cutslab, generate_slabs_for_miller
    from taskerslabgen.advanced import (
        assign_plane_names,
        build_adjacency_matrix,
        build_surface,
        compute_reduced_counts,
        identify_planes,
    )

    assert callable(generate_slabs_for_miller)
    assert callable(cutslab)
    assert callable(build_surface)
    assert callable(identify_planes)
    assert callable(assign_plane_names)
    assert callable(build_adjacency_matrix)
    assert callable(compute_reduced_counts)


def test_quiet_by_default_no_plot_side_effects(ceo2_bulk, tmp_path, capsys):
    from taskerslabgen import generate_slabs_for_miller

    result = generate_slabs_for_miller(
        ceo2_bulk,
        CEO2_CHARGES,
        millers=(1, 1, 0),
        layer_thickness_list=[2],
        bulk_name="CeO2",
        vacuum=15.0,
        plot_out_dir=tmp_path.as_posix(),
        verbose=False,
        candidates="best",
    )
    assert (1, 1, 0) in result
    assert list(tmp_path.glob("*.png")) == []
    captured = capsys.readouterr()
    assert "Reconstructing Tasker III" not in captured.out


# ------------------------------------------------------------------
# 2. Tasker III genslab roundtrip – CeO2 (001)
# ------------------------------------------------------------------
def test_genslab_tasker3_ceo2_001(ceo2_bulk):
    from taskerslabgen import generate_slabs_for_miller

    result = generate_slabs_for_miller(
        ceo2_bulk,
        CEO2_CHARGES,
        millers=(0, 0, 1),
        layer_thickness_list=[2],
        bulk_name="CeO2",
        vacuum=15.0,
        plot=False,
        verbose=False,
        bond_distances=BOND_DISTS,
        candidates="best",
        prefer_plane="O",
    )

    assert (0, 0, 1) in result
    terminations = result[(0, 0, 1)]
    assert len(terminations) >= 1
    for tid, info in terminations.items():
        assert info["tasker_type"] == "III"
        assert len(info["atoms"]) >= 1
        slab = info["atoms"][0]
        assert len(slab) > 0
        assert info["reconstruction"] is not None
        assert abs(_formal_charge_sum(slab, CEO2_CHARGES)) < 1e-6


# ------------------------------------------------------------------
# 3. Tasker I/II genslab roundtrip – CeO2 (110)
# ------------------------------------------------------------------
def test_genslab_tasker12_ceo2_110(ceo2_bulk):
    from taskerslabgen import generate_slabs_for_miller

    result = generate_slabs_for_miller(
        ceo2_bulk,
        CEO2_CHARGES,
        millers=(1, 1, 0),
        layer_thickness_list=[2],
        bulk_name="CeO2",
        vacuum=15.0,
        plot=False,
        verbose=False,
        candidates="best",
    )

    assert (1, 1, 0) in result
    terminations = result[(1, 1, 0)]
    assert len(terminations) >= 1
    for tid, info in terminations.items():
        assert info["tasker_type"] == "I/II"
        slab = info["atoms"][0]
        assert len(slab) > 0
        assert abs(_formal_charge_sum(slab, CEO2_CHARGES)) < 1e-6
        cand = info["candidate"]
        assert abs(cand["net_dipole"]) <= 1e-6


# ------------------------------------------------------------------
# 3b. Tasker I/II genslab – IrO2 (110)
# ------------------------------------------------------------------
def test_genslab_tasker12_iro2_110(iro2_bulk):
    from taskerslabgen import generate_slabs_for_miller

    result = generate_slabs_for_miller(
        iro2_bulk,
        IRO2_CHARGES,
        millers=(1, 1, 0),
        layer_thickness_list=[3],
        bulk_name="IrO2",
        vacuum=15.0,
        plot=False,
        verbose=False,
        candidates="best",
    )

    assert (1, 1, 0) in result
    terminations = result[(1, 1, 0)]
    assert len(terminations) >= 1
    for tid, info in terminations.items():
        assert info["tasker_type"] == "I/II"
        slab = info["atoms"][0]
        assert len(slab) > 0
        assert abs(_formal_charge_sum(slab, IRO2_CHARGES)) < 1e-6


# ------------------------------------------------------------------
# 4. cutslab on thick slab – expected sub-slabs
# ------------------------------------------------------------------
def test_cutslab_produces_subslabs(ceo2_bulk):
    from taskerslabgen import cutslab, generate_slabs_for_miller

    result = generate_slabs_for_miller(
        ceo2_bulk,
        CEO2_CHARGES,
        millers=(1, 1, 0),
        layer_thickness_list=[3],
        bulk_name="CeO2",
        vacuum=15.0,
        plot=False,
        verbose=False,
        candidates="best",
    )

    terminations = result[(1, 1, 0)]
    tid = min(terminations.keys())
    thick_slab = terminations[tid]["atoms"][0]

    sub_slabs = cutslab(
        thick_slab,
        CEO2_CHARGES,
        axis=2,
        plot=False,
        cut_at="termination",
        cuts="right",
        vacuum=15.0,
    )

    assert len(sub_slabs) >= 1
    for slab in sub_slabs:
        assert "cut_bottom_plane" in slab.info
        assert "cut_top_plane" in slab.info
        assert "cut_n_planes" in slab.info
        assert abs(_formal_charge_sum(slab, CEO2_CHARGES)) < 1e-6

    sizes = [len(s) for s in sub_slabs]
    assert sizes == sorted(sizes), "Sub-slabs should be sorted by size"


# ------------------------------------------------------------------
# 5. assign_plane_names fingerprinting (stacking alternation for 110)
# ------------------------------------------------------------------
def test_assign_plane_names_110_alternation(ceo2_bulk):
    from taskerslabgen import plane_name_base
    from taskerslabgen.advanced import (
        assign_plane_names,
        build_surface,
        compute_projection,
        identify_planes,
    )

    surf = build_surface(ceo2_bulk, (1, 1, 0), layers=1, vacuum=0.0)
    atoms_z, L = compute_projection(ceo2_bulk, surf, CEO2_CHARGES, (1, 1, 0))
    planes = identify_planes(atoms_z, L, plane_tol=None)
    planes_sorted = sorted(planes, key=lambda p: p["z_center"] % L)

    names, name_map = assign_plane_names(planes_sorted, atoms=surf)

    unique_names = set(names)
    assert len(unique_names) <= 2, (
        f"CeO2 (110) should have at most 2 plane labels, got {unique_names}"
    )
    bases = {plane_name_base(n) for n in names}
    assert len(bases) <= 2
    if len(names) >= 4:
        assert names[0] == names[2], (
            f"Expected ABAB alternation, got {names}"
        )


def test_iro2_001_plane_variants(iro2_bulk):
    from taskerslabgen import plane_name_base
    from taskerslabgen.advanced import (
        assign_plane_names,
        build_surface,
        compute_projection,
        identify_planes,
    )

    surf = build_surface(iro2_bulk, (0, 0, 1), layers=1, vacuum=0.0)
    atoms_z, L = compute_projection(iro2_bulk, surf, IRO2_CHARGES, (0, 0, 1))
    planes = sorted(
        identify_planes(atoms_z, L, plane_tol=None),
        key=lambda p: p["z_center"] % L,
    )
    names, _ = assign_plane_names(planes, atoms=surf)

    # Mirror-related IrO2 planes: same composition, two geometric variants.
    assert len(names) == 2
    assert sorted(names) == ["IrO2-a", "IrO2-b"]
    assert plane_name_base(names[0]) == "IrO2"
    assert plane_name_base(names[1]) == "IrO2"


def test_iro2_001_fixed_plane_tol_override(iro2_bulk):
    from taskerslabgen.advanced import build_surface, compute_projection, identify_planes

    surf = build_surface(iro2_bulk, (0, 0, 1), layers=1, vacuum=0.0)
    atoms_z, L = compute_projection(iro2_bulk, surf, IRO2_CHARGES, (0, 0, 1))
    planes = identify_planes(atoms_z, L, plane_tol=0.05)
    assert len(planes) == 2


def test_albite_001_stays_finely_cut(albite_bulk):
    from taskerslabgen.advanced import build_surface, compute_projection, identify_planes

    surf = build_surface(albite_bulk, (0, 0, 1), layers=1, vacuum=0.0)
    atoms_z, L = compute_projection(albite_bulk, surf, ALBITE_CHARGES, (0, 0, 1))
    planes = identify_planes(atoms_z, L, plane_tol=None)
    sizes = [len(p["indices"]) for p in planes]
    assert len(planes) >= 6, (
        f"Albite (001) should stay finely cut, got {len(planes)} planes"
    )
    assert max(sizes) <= 6, (
        f"Albite (001) should not collapse into thick sheets, "
        f"max atoms/plane={max(sizes)}"
    )


def test_plane_name_matches_semantics():
    from taskerslabgen import plane_name_base, plane_name_matches

    assert plane_name_base("IrO2-a") == "IrO2"
    assert plane_name_base("IrO2-a-recon") == "IrO2"
    assert plane_name_base("O4-recon") == "O4"
    assert plane_name_base("Ce2O4") == "Ce2O4"

    assert plane_name_matches("IrO2", "IrO2-a")
    assert plane_name_matches("IrO2", "IrO2-b")
    assert plane_name_matches("IrO2", "IrO2-a-recon")
    assert plane_name_matches("IrO2-a", "IrO2-a-recon")
    assert plane_name_matches("O4", "O4-recon")
    assert plane_name_matches("O4-recon", "O4-recon")
    assert not plane_name_matches("IrO2-a", "IrO2-b")
    assert not plane_name_matches("O4", "O2")


def test_prefer_plane_type_matches_variants(iro2_bulk):
    from taskerslabgen import generate_slabs_for_miller, plane_name_base

    result = generate_slabs_for_miller(
        iro2_bulk,
        IRO2_CHARGES,
        millers=(0, 0, 1),
        layer_thickness_list=[2],
        bulk_name="IrO2",
        vacuum=15.0,
        plot=False,
        verbose=False,
        candidates="all",
        prefer_plane="IrO2",
    )
    terminations = result[(0, 0, 1)]
    assert len(terminations) >= 1
    for info in terminations.values():
        assert plane_name_base(info["plane_type"]) == "IrO2"


# ------------------------------------------------------------------
# 6. prefer_plane exclusive matching
# ------------------------------------------------------------------
def test_prefer_plane_exclusive_element_matching(ceo2_bulk):
    from taskerslabgen import generate_slabs_for_miller

    result = generate_slabs_for_miller(
        ceo2_bulk,
        CEO2_CHARGES,
        millers=(0, 0, 1),
        layer_thickness_list=[2],
        bulk_name="CeO2",
        vacuum=15.0,
        plot=False,
        verbose=False,
        bond_distances=BOND_DISTS,
        candidates="all",
        prefer_plane="O",
    )

    terminations = result[(0, 0, 1)]
    for tid, info in terminations.items():
        counts = info["plane_counts"]
        present_elements = {Z for Z, c in counts.items() if c > 0}
        from ase.data import atomic_numbers

        assert present_elements == {atomic_numbers["O"]}, (
            f"prefer_plane='O' should only select pure-O planes, "
            f"got elements Z={present_elements}"
        )


# ------------------------------------------------------------------
# 7. Stoichiometry helpers
# ------------------------------------------------------------------
def test_reduced_counts_and_stoichiometry(ceo2_bulk):
    from taskerslabgen import generate_slabs_for_miller
    from taskerslabgen.advanced import compute_reduced_counts, is_stoichiometric_sequence
    from collections import Counter

    atoms_z = np.array(
        [
            [58, 0.0, 4.0],
            [8, 0.0, -2.0],
            [8, 0.0, -2.0],
        ]
    )
    reduced = compute_reduced_counts(atoms_z)
    assert reduced[58] == 1
    assert reduced[8] == 2

    ok, k = is_stoichiometric_sequence({58: 2, 8: 4}, reduced)
    assert ok and k == 2

    result = generate_slabs_for_miller(
        ceo2_bulk,
        CEO2_CHARGES,
        millers=(1, 1, 0),
        layer_thickness_list=[2],
        bulk_name="CeO2",
        vacuum=15.0,
        plot=False,
        verbose=False,
        candidates="best",
    )
    slab = next(iter(result[(1, 1, 0)].values()))["atoms"][0]
    counts = Counter(int(z) for z in slab.numbers)
    # Bulk formula unit CeO2 → reduced {58:1, 8:2}
    assert _stoich_ratio(slab, {58: 1, 8: 2}) is not None
    assert abs(_formal_charge_sum(slab, CEO2_CHARGES)) < 1e-6
    assert sum(counts.values()) == len(slab)
