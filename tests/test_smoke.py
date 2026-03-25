"""
Smoke tests for taskerslabgen.

These are lightweight integration tests that verify the main workflows
run end-to-end without error and produce sensible outputs.  They use
the CeO2 bulk files shipped in the repository.
"""
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from ase.io import read
from ase.build import bulk as ase_bulk

BULK_DIR = Path(__file__).resolve().parent.parent / "bulk_files"
CEO2_PATH = BULK_DIR / "CeO2_fluorite.cif"
CEO2_CHARGES = {"Ce": 4.0, "O": -2.0}
BOND_DISTS = {"Ce-Ce": None, "O-O": None, "Ce-O": 2.35}


@pytest.fixture
def ceo2_bulk():
    return read(CEO2_PATH.as_posix())


# ------------------------------------------------------------------
# 1. Import check
# ------------------------------------------------------------------
def test_import_public_api():
    from taskerslabgen import (
        generate_slabs_for_miller,
        cutslab,
        build_surface,
        identify_planes,
        assign_plane_names,
        compute_reduced_counts,
        build_adjacency_matrix,
    )
    assert callable(generate_slabs_for_miller)
    assert callable(cutslab)


# ------------------------------------------------------------------
# 2. Tasker III genslab roundtrip – CeO2 (001)
# ------------------------------------------------------------------
def test_genslab_tasker3_ceo2_001(ceo2_bulk):
    from taskerslabgen import generate_slabs_for_miller

    with tempfile.TemporaryDirectory() as tmpdir:
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


# ------------------------------------------------------------------
# 4. cutslab on thick slab – expected sub-slabs
# ------------------------------------------------------------------
def test_cutslab_produces_subslabs(ceo2_bulk):
    from taskerslabgen import generate_slabs_for_miller, cutslab

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

    with tempfile.TemporaryDirectory() as tmpdir:
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

    sizes = [len(s) for s in sub_slabs]
    assert sizes == sorted(sizes), "Sub-slabs should be sorted by size"


# ------------------------------------------------------------------
# 5. assign_plane_names fingerprinting (P0 P1 alternation for 110)
# ------------------------------------------------------------------
def test_assign_plane_names_110_alternation(ceo2_bulk):
    from taskerslabgen import (
        build_surface,
        compute_projection,
        identify_planes,
        assign_plane_names,
    )

    surf = build_surface(ceo2_bulk, (1, 1, 0), layers=1, vacuum=0.0)
    atoms_z, L = compute_projection(ceo2_bulk, surf, CEO2_CHARGES, (1, 1, 0))
    planes = identify_planes(atoms_z, L, plane_tol=0.05)
    planes_sorted = sorted(planes, key=lambda p: p["z_center"] % L)

    names, name_map = assign_plane_names(planes_sorted, atoms=surf)

    unique_names = set(names)
    assert len(unique_names) <= 2, (
        f"CeO2 (110) should have at most 2 plane types, got {unique_names}"
    )
    if len(names) >= 4:
        assert names[0] == names[2], (
            f"Expected ABAB alternation, got {names}"
        )


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
