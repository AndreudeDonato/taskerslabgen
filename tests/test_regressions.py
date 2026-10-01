"""
Regression tests for the issues found in the 0.3.1 review (REVIEW_NOTES.md).

Each test names the review item it covers.  Structures are unit cells (at
most 2x2 in plane) so the Tasker III enumeration stays small.
"""
import warnings
from collections import Counter
from functools import reduce
from math import gcd
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk, surface
from ase.data import covalent_radii
from ase.io import read
from ase.neighborlist import neighbor_list

BULK_DIR = Path(__file__).resolve().parent.parent / "bulk_files"
CEO2 = read((BULK_DIR / "CeO2_fluorite.cif").as_posix())
IRO2 = read((BULK_DIR / "IrO2_rutile.cif").as_posix())
ZNO = bulk("ZnO", "wurtzite", a=3.25, c=5.21, u=0.382)
GAN = bulk("GaN", "wurtzite", a=3.19, c=5.19, u=0.377)
MGO = bulk("MgO", "rocksalt", a=4.21, cubic=True)

Q_CEO2 = {"Ce": 4.0, "O": -2.0}
Q_IRO2 = {"Ir": 4.0, "O": -2.0}
Q_ZNO = {"Zn": 2.0, "O": -2.0}
Q_GAN = {"Ga": 3.0, "N": -3.0}
Q_MGO = {"Mg": 2.0, "O": -2.0}
BOND_DISTS_CEO2 = {"Ce-Ce": None, "O-O": None, "Ce-O": 2.35}


# ------------------------------------------------------------------
# helpers (independent of the library internals)
# ------------------------------------------------------------------
def _tasker_type(atoms, charges, hkl, plane_tol=None):
    from taskerslabgen import (
        build_surface,
        compute_projection,
        compute_reduced_counts,
        enumerate_cut_pairs,
        identify_planes,
        select_best_sequence,
    )

    surf = build_surface(atoms, hkl, layers=1)
    atoms_z, L = compute_projection(atoms, surf, charges, hkl)
    planes = identify_planes(atoms_z, L, plane_tol=plane_tol)
    best = select_best_sequence(
        enumerate_cut_pairs(planes, L, compute_reduced_counts(atoms_z))
    )
    assert best is not None
    return "I/II" if best["is_tasker_ii"] else "III"


def _reduced(atoms):
    counts = Counter(int(z) for z in atoms.numbers)
    g = reduce(gcd, counts.values())
    return {z: c // g for z, c in counts.items()}


def _bulk_max_gap(atoms, hkl):
    """Largest atom-free z-gap of the bulk along the surface normal."""
    s = surface(atoms, hkl, layers=1)
    L = atoms.get_volume() / np.linalg.norm(np.cross(s.cell[0], s.cell[1]))
    z = np.sort(s.positions[:, 2] % L)
    return float(np.max(np.diff(np.concatenate([z, [z[0] + L]]))))


def _assert_valid_slab(slab, charges, reduced, max_gap=None):
    counts = Counter(int(z) for z in slab.numbers)
    ks = {counts.get(z, 0) / r for z, r in reduced.items()}
    assert len(ks) == 1 and next(iter(ks)) >= 1 and float(next(iter(ks))).is_integer(), (
        f"non-stoichiometric slab {slab.get_chemical_formula()}"
    )
    q = np.array([charges[s] for s in slab.get_chemical_symbols()])
    assert abs(q.sum()) < 1e-6, f"charged slab {slab.get_chemical_formula()} Q={q.sum():+.3f}"
    z = slab.positions[:, 2]
    mu = float(np.sum(q * (z - z.mean())))
    assert abs(mu) < 1e-4, f"polar slab {slab.get_chemical_formula()} mu={mu:+.4f}"
    if max_gap is not None:
        gap = float(np.max(np.diff(np.sort(z))))
        assert gap <= max_gap + 1e-3, (
            f"slab {slab.get_chemical_formula()} has an internal gap of {gap:.2f} A "
            f"(bulk max {max_gap:.2f} A)"
        )


def _shifted(atoms, shift):
    out = atoms.copy()
    out.set_scaled_positions((out.get_scaled_positions() + shift) % 1.0)
    return out


# ------------------------------------------------------------------
# C1: plane clustering must classify textbook surfaces correctly
# ------------------------------------------------------------------
@pytest.mark.parametrize(
    "atoms, charges, hkl, expected",
    [
        (ZNO, Q_ZNO, (0, 0, 1), "III"),
        (GAN, Q_GAN, (0, 0, 1), "III"),
        (ZNO, Q_ZNO, (1, 0, 0), "I/II"),
        (MGO, Q_MGO, (1, 1, 1), "III"),
        (MGO, Q_MGO, (1, 0, 0), "I/II"),
        (IRO2, Q_IRO2, (1, 1, 0), "I/II"),
        (IRO2, Q_IRO2, (1, 0, 0), "I/II"),
        (CEO2, Q_CEO2, (1, 1, 1), "I/II"),
        (CEO2, Q_CEO2, (1, 1, 0), "I/II"),
        (CEO2, Q_CEO2, (0, 0, 1), "III"),
    ],
    ids=["ZnO0001", "GaN0001", "ZnO10-10", "MgO111", "MgO100", "IrO2110",
         "IrO2100", "CeO2111", "CeO2110", "CeO2001"],
)
def test_known_tasker_classification(atoms, charges, hkl, expected):
    assert _tasker_type(atoms, charges, hkl) == expected


@pytest.mark.parametrize(
    "atoms, charges, hkl",
    [(CEO2, Q_CEO2, (1, 1, 1)), (IRO2, Q_IRO2, (1, 0, 0)), (IRO2, Q_IRO2, (1, 1, 0))],
    ids=["CeO2111", "IrO2100", "IrO2110"],
)
def test_result_independent_of_bulk_origin(atoms, charges, hkl):
    from taskerslabgen import generate_slabs_for_miller

    rng = np.random.default_rng(0)
    seen = set()
    for shift in [np.zeros(3)] + [rng.random(3) for _ in range(4)]:
        info = next(iter(
            generate_slabs_for_miller(_shifted(atoms, shift), charges, hkl, [2])[hkl].values()
        ))
        seen.add((info["tasker_type"], info["atoms"][0].get_chemical_formula()))
    assert len(seen) == 1, f"origin-dependent result: {seen}"


# ------------------------------------------------------------------
# C2 + C5 + C6: cutslab must peel a vacuum slab into a full series
# ------------------------------------------------------------------
@pytest.fixture(scope="module")
def ceo2_111_slab():
    from taskerslabgen import generate_slabs_for_miller

    res = generate_slabs_for_miller(CEO2, Q_CEO2, (1, 1, 1), [3])
    return next(iter(res[(1, 1, 1)].values()))["atoms"][0]


def test_cutslab_termination_series(ceo2_111_slab):
    from taskerslabgen import cutslab

    subs = cutslab(ceo2_111_slab, Q_CEO2)
    assert [s.get_chemical_formula() for s in subs] == ["Ce4O8", "Ce8O16", "Ce12O24"]


def test_cutslab_all_mode_has_no_glued_or_duplicate_slabs(ceo2_111_slab):
    from taskerslabgen import cutslab

    max_gap = _bulk_max_gap(CEO2, (1, 1, 1))
    reduced = _reduced(CEO2)
    subs = cutslab(ceo2_111_slab, Q_CEO2, cut_at="all", cuts="all")
    spans = [(s.info["cut_bottom_idx"], s.info["cut_top_idx"]) for s in subs]
    assert len(spans) == len(set(spans)), f"duplicate sub-slabs: {spans}"
    for s in subs:
        _assert_valid_slab(s, Q_CEO2, reduced, max_gap)


def test_cutslab_all_mode_right_anchors_at_bottom_plane(ceo2_111_slab):
    from taskerslabgen import cutslab

    subs = cutslab(ceo2_111_slab, Q_CEO2, cut_at="all", cuts="right")
    assert {s.info["cut_bottom_idx"] for s in subs} == {0}
    assert [s.get_chemical_formula() for s in subs] == ["Ce4O8", "Ce8O16", "Ce12O24"]


def test_cutslab_tasker3_series_keeps_reconstruction():
    from taskerslabgen import cutslab, generate_slabs_for_miller

    res = generate_slabs_for_miller(
        CEO2, Q_CEO2, (0, 0, 1), [4], prefer_plane="O", bond_distances=BOND_DISTS_CEO2
    )
    term = next(iter(res[(0, 0, 1)].values()))
    subs = cutslab(term["atoms"][0], Q_CEO2, reconstruction=term["reconstruction"])
    assert [s.get_chemical_formula() for s in subs] == [
        f"Ce{2 * m}O{4 * m}" for m in range(1, 9)
    ]
    reduced = _reduced(CEO2)
    for s in subs:
        _assert_valid_slab(s, Q_CEO2, reduced)


# ------------------------------------------------------------------
# C3: every returned slab is stoichiometric, neutral, non-polar, contiguous
# ------------------------------------------------------------------
@pytest.mark.parametrize(
    "atoms, charges, hkl, kwargs",
    [
        (IRO2, Q_IRO2, (1, 0, 0), {}),
        (IRO2, Q_IRO2, (1, 1, 0), {}),
        (IRO2, Q_IRO2, (1, 0, 1), {}),
        (CEO2, Q_CEO2, (1, 1, 1), {}),
        (_shifted(CEO2, [0.31, 0.77, 0.12]), Q_CEO2, (1, 1, 1), {}),
        (CEO2, Q_CEO2, (1, 1, 0), {}),
        (CEO2, Q_CEO2, (0, 0, 1), {"bond_distances": BOND_DISTS_CEO2}),
        (MGO, Q_MGO, (1, 1, 1), {}),
    ],
    ids=["IrO2100", "IrO2110", "IrO2101", "CeO2111", "CeO2111shift", "CeO2110",
         "CeO2001", "MgO111"],
)
def test_every_generated_slab_is_valid(atoms, charges, hkl, kwargs):
    from taskerslabgen import generate_slabs_for_miller

    res = generate_slabs_for_miller(atoms, charges, hkl, [1, 2, 3], candidates="all", **kwargs)
    reduced = _reduced(atoms)
    max_gap = _bulk_max_gap(atoms, hkl)
    assert res[hkl]
    for info in res[hkl].values():
        for slab in info["atoms"]:
            _assert_valid_slab(slab, charges, reduced, max_gap)
            assert not any(k.startswith("_tsg") for k in slab.arrays), "internal array leaked"


def test_polar_reconstruction_is_rejected_not_returned():
    """Wurtzite (0001) cannot be fixed by symmetric deletion from one plane type."""
    from taskerslabgen import generate_slabs_for_miller

    with pytest.raises(ValueError, match="(?i)dipole"):
        generate_slabs_for_miller(ZNO * (2, 2, 1), Q_ZNO, (0, 0, 1), [2])


def test_odd_excess_error_suggests_supercell():
    from taskerslabgen import generate_slabs_for_miller

    with pytest.raises(ValueError, match="(?i)supercell"):
        generate_slabs_for_miller(ZNO, Q_ZNO, (0, 0, 1), [2])


# ------------------------------------------------------------------
# C4: adjacency must use the true bulk lattice in the surface frame
# ------------------------------------------------------------------
def _reference_bond_counts(atoms, hkl, lo=0.85, hi=1.15, n_layers=5):
    """Bonds between each unit-cell atom and all copies of every other atom,
    counted in the middle layer of a thick slab (stacked with the true a3)."""
    n = len(atoms)
    thick = surface(atoms, hkl, layers=n_layers, vacuum=10.0)
    thick.pbc = (True, True, False)
    radii = covalent_radii[thick.numbers]
    cutoff = hi * 2.0 * float(np.max(covalent_radii[atoms.numbers]))
    i_idx, j_idx, d = neighbor_list("ijd", thick, cutoff)
    ref = radii[i_idx] + radii[j_idx]
    ok = (d >= lo * ref) & (d <= hi * ref)
    mid = n_layers // 2
    in_mid = (i_idx // n) == mid
    counts = np.zeros((n, n), dtype=int)
    np.add.at(counts, (i_idx[ok & in_mid] % n, j_idx[ok & in_mid] % n), 1)
    return counts


@pytest.mark.parametrize(
    "atoms, hkl",
    [(CEO2, (1, 1, 0)), (CEO2, (1, 1, 1)), (CEO2, (0, 0, 1)), (IRO2, (1, 1, 0)), (IRO2, (1, 0, 1))],
    ids=["CeO2110", "CeO2111", "CeO2001", "IrO2110", "IrO2101"],
)
def test_adjacency_matches_true_bulk_bonding(atoms, hkl):
    from taskerslabgen import build_adjacency_matrix, build_surface

    surf = build_surface(atoms, hkl, layers=1)
    adj = build_adjacency_matrix(surf, bulk_atoms=atoms, miller=hkl)
    np.testing.assert_array_equal(np.asarray(adj, dtype=int), _reference_bond_counts(atoms, hkl))


def test_adjacency_with_bulk_requires_miller():
    from taskerslabgen import build_adjacency_matrix, build_surface

    surf = build_surface(CEO2, (1, 1, 1), layers=1)
    with pytest.raises(ValueError, match="miller"):
        build_adjacency_matrix(surf, bulk_atoms=CEO2)


# ------------------------------------------------------------------
# C6: plane names must not depend on in-plane translation
# ------------------------------------------------------------------
def test_translated_identical_planes_share_a_name(ceo2_111_slab):
    from taskerslabgen import assign_plane_names, identify_planes
    from taskerslabgen.core import _charges_to_list

    slab = ceo2_111_slab
    atoms_z = np.column_stack([
        slab.numbers, slab.positions[:, 2], _charges_to_list(slab, Q_CEO2)
    ])
    planes = sorted(identify_planes(atoms_z, slab.cell[2, 2]), key=lambda p: p["z_center"])
    names, _ = assign_plane_names(planes, atoms=slab)
    by_comp = {}
    for p, name in zip(planes, names):
        by_comp.setdefault(tuple(sorted(p["counts"].items())), set()).add(name)
    assert all(len(v) == 1 for v in by_comp.values()), f"names: {names}"


# ------------------------------------------------------------------
# N1: labels depend only on the plane, so genslab and cutslab agree
# ------------------------------------------------------------------
@pytest.mark.parametrize(
    "atoms, charges, hkl, kwargs",
    [
        (CEO2, Q_CEO2, (1, 1, 1), {}),
        (IRO2, Q_IRO2, (1, 1, 0), {}),
        (IRO2, Q_IRO2, (0, 0, 1), {}),
        (CEO2, Q_CEO2, (0, 0, 1), {"prefer_plane": "O", "bond_distances": BOND_DISTS_CEO2}),
    ],
    ids=["CeO2111", "IrO2110", "IrO2001", "CeO2001recon"],
)
def test_genslab_label_selects_same_planes_in_cutslab(atoms, charges, hkl, kwargs):
    from taskerslabgen import cutslab, generate_slabs_for_miller

    res = generate_slabs_for_miller(atoms, charges, hkl, [3], candidates="all", **kwargs)
    for term in res[hkl].values():
        slab, label = term["atoms"][0], term["plane_type"]
        recon = term["reconstruction"]
        # cutslab gives the slab's bottom plane the label genslab reported ...
        by_termination = cutslab(slab, charges, reconstruction=recon)
        assert {s.info["cut_bottom_plane"] for s in by_termination} == {label}
        # ... so genslab's label can be passed straight to cut_at.
        by_label = cutslab(slab, charges, cut_at=label, reconstruction=recon)
        assert by_label
        assert {s.info["cut_bottom_plane"] for s in by_label} == {label}


@pytest.mark.parametrize(
    "atoms, charges, hkl",
    [(CEO2, Q_CEO2, (1, 1, 1)), (IRO2, Q_IRO2, (1, 1, 0)), (IRO2, Q_IRO2, (0, 0, 1))],
    ids=["CeO2111", "IrO2110", "IrO2001"],
)
def test_plane_labels_independent_of_bulk_origin(atoms, charges, hkl):
    from taskerslabgen import generate_slabs_for_miller

    rng = np.random.default_rng(3)
    seen = set()
    for shift in [np.zeros(3)] + [rng.random(3) for _ in range(4)]:
        res = generate_slabs_for_miller(_shifted(atoms, shift), charges, hkl, [2], candidates="all")
        seen.add(frozenset(info["plane_type"] for info in res[hkl].values()))
    assert len(seen) == 1, f"origin-dependent labels: {seen}"


# ------------------------------------------------------------------
# C9 / C10: error messages and ASE deprecations
# ------------------------------------------------------------------
def test_cutslab_reports_empty_result_not_unknown_mode(ceo2_111_slab):
    from taskerslabgen import cutslab

    noisy = ceo2_111_slab.copy()
    noisy.positions[:, 2] += np.random.default_rng(1).normal(0.0, 0.02, len(noisy))
    with pytest.raises(ValueError, match="No stoichiometric"):
        cutslab(noisy, Q_CEO2, plane_tol=0.3, dipole_tol=1e-6)
    with pytest.raises(ValueError, match="Unknown cuts mode"):
        cutslab(ceo2_111_slab, Q_CEO2, cuts="sideways")


def test_cutslab_on_polar_slab_points_to_reconstruction():
    """A polar slab has no zero-dipole cut; cutslab must say so, not fake a fix."""
    from taskerslabgen import cutslab

    polar = surface(CEO2, (0, 0, 1), layers=3, vacuum=10.0)
    with pytest.raises(ValueError, match="generate_slabs_for_miller"):
        cutslab(polar, Q_CEO2, cut_at="all")


# ------------------------------------------------------------------
# C8 / N2: dipole_tol is per formula unit (default 0.05)
# ------------------------------------------------------------------
def test_cif_rounded_coordinates_stay_non_polar():
    """4-decimal CIF coordinates give a ~6e-4 e*A noise dipole (failed 1e-6)."""
    rounded = IRO2.copy()
    rounded.set_scaled_positions(np.round(rounded.get_scaled_positions() + 0.123456, 4))
    assert _tasker_type(rounded, Q_IRO2, (1, 1, 0)) == "I/II"


def test_dipole_tol_is_per_formula_unit():
    from taskerslabgen import select_best_sequence

    def seq(mu, k):
        return {"is_neutral": True, "is_stoich": True, "is_full_period": True,
                "net_dipole": mu, "stoich_k": k, "dipole_per_fu": abs(mu) / k,
                "bottom_cut": 0}

    assert select_best_sequence([seq(0.16, 4)])["is_tasker_ii"]       # 0.04 per f.u.
    assert not select_best_sequence([seq(0.16, 2)])["is_tasker_ii"]   # 0.08 per f.u.


def test_cutslab_relaxed_slab_keeps_thick_cuts(ceo2_111_slab):
    """Relaxation dipoles do not grow with thickness; per-f.u. tolerance keeps the series."""
    from taskerslabgen import cutslab

    relaxed = ceo2_111_slab.copy()
    z = relaxed.positions[:, 2]
    relaxed.positions[z < z.min() + 1.0, 2] += 0.10   # outer planes relax inward
    relaxed.positions[z > z.max() - 1.0, 2] -= 0.10
    relaxed.positions[:, 2] += np.random.default_rng(0).normal(0.0, 0.01, len(relaxed))
    subs = cutslab(relaxed, Q_CEO2, dipole_tol=0.3)
    assert [s.get_chemical_formula() for s in subs] == ["Ce4O8", "Ce8O16", "Ce12O24"]


# ------------------------------------------------------------------
# N3: best Tasker I/II termination = fewest broken bonds, origin-independent
# ------------------------------------------------------------------
ALBITE = read((BULK_DIR / "NaAlSi3O8_albite.cif").as_posix())
Q_ALBITE = {"Na": 1.0, "Al": 3.0, "Si": 4.0, "O": -2.0}


def _srtio3():
    from ase.spacegroup import crystal

    return crystal(["Sr", "Ti", "O"], [(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
                   spacegroup=221, cellpar=[3.905] * 3 + [90] * 3)


@pytest.mark.parametrize(
    "atoms, charges, hkl",
    [(ALBITE, Q_ALBITE, (0, 0, 1)), (_srtio3(), {"Sr": 2.0, "Ti": 4.0, "O": -2.0}, (1, 0, 0)),
     (IRO2, Q_IRO2, (1, 1, 1))],
    ids=["albite001", "SrTiO3100", "IrO2111"],
)
def test_best_termination_independent_of_bulk_origin(atoms, charges, hkl):
    from taskerslabgen import generate_slabs_for_miller

    rng = np.random.default_rng(1)
    best = set()
    for shift in [np.zeros(3)] + [rng.random(3) for _ in range(5)]:
        res = generate_slabs_for_miller(_shifted(atoms, shift), charges, hkl, [2])
        best.add(next(iter(res[hkl].values()))["plane_type"])
    assert len(best) == 1, f"origin-dependent best termination: {best}"


def test_terminations_ranked_by_broken_bonds():
    from taskerslabgen import generate_slabs_for_miller

    res = generate_slabs_for_miller(ALBITE, Q_ALBITE, (0, 0, 1), [2], candidates="all")
    ranked = [res[(0, 0, 1)][tid] for tid in sorted(res[(0, 0, 1)])]
    bonds = [t["candidate"]["broken_bonds"] for t in ranked]
    assert bonds == sorted(bonds) and bonds[0] < bonds[-1]
    assert ranked[0]["plane_type"] == "Si"   # the Si cut breaks 4 bonds, the O cut 9


# ------------------------------------------------------------------
# Step 4: cutslab surface descriptor matched to the bulk (bulk_atoms=)
# ------------------------------------------------------------------
def _relax_surfaces(slab, kind):
    """Distort the four O of each surface plane of a CeO2(111) slab."""
    out = slab.copy()
    z = out.positions[:, 2]
    oxygens = np.flatnonzero(out.numbers == 8)
    for top in (False, True):
        order = np.argsort(z[oxygens])
        idx = oxygens[order[-4:]] if top else oxygens[order[:4]]
        if kind == "rumpled":            # 2 of 4 O move outwards by 0.15 A
            out.positions[idx[:2], 2] += 0.15 if top else -0.15
        elif kind == "shifted":          # 2 of 4 O shift in-plane by 1.0 A
            out.positions[idx[:2], 0] += 1.0
        elif kind == "outward":          # whole plane relaxes outwards by 0.2 A
            out.positions[idx, 2] += 0.2 if top else -0.2
    return out


def test_bulk_matching_keeps_rumpled_surface_plane_whole(ceo2_111_slab):
    from taskerslabgen import cutslab

    rumpled = _relax_surfaces(ceo2_111_slab, "rumpled")
    subs = cutslab(rumpled, Q_CEO2, bulk_atoms=CEO2, dipole_tol=0.3)
    assert [s.get_chemical_formula() for s in subs] == ["Ce4O8", "Ce8O16", "Ce12O24"]
    assert {(s.info["cut_bottom_plane"], s.info["cut_top_plane"]) for s in subs} == {("O4", "O4")}


def test_deformed_surface_plane_gets_primed_label(ceo2_111_slab):
    from taskerslabgen import cutslab

    shifted = _relax_surfaces(ceo2_111_slab, "shifted")
    subs = cutslab(shifted, Q_CEO2, bulk_atoms=CEO2)
    assert [s.get_chemical_formula() for s in subs] == ["Ce4O8", "Ce8O16", "Ce12O24"]
    surfaces = [(s.info["cut_bottom_plane"], s.info["cut_top_plane"]) for s in subs]
    assert surfaces == [("O4'", "O4"), ("O4'", "O4"), ("O4'", "O4'")]


def test_cut_dipoles_use_relaxed_positions(ceo2_111_slab):
    """The thinnest cut keeps one relaxed surface: 0.4 e*A per formula unit."""
    from taskerslabgen import cutslab

    relaxed = _relax_surfaces(ceo2_111_slab, "outward")
    subs = cutslab(relaxed, Q_CEO2, bulk_atoms=CEO2, dipole_tol=0.3)
    assert [s.get_chemical_formula() for s in subs] == ["Ce8O16", "Ce12O24"]


@pytest.mark.parametrize("hkl", [(0, 0, 1), (1, 1, 0)], ids=["IrO2001", "IrO2110"])
def test_bulk_matched_labels_agree_with_genslab(hkl):
    from taskerslabgen import cutslab, generate_slabs_for_miller

    for term in generate_slabs_for_miller(IRO2, Q_IRO2, hkl, [3], candidates="all")[hkl].values():
        subs = cutslab(term["atoms"][0], Q_IRO2, bulk_atoms=IRO2)
        assert {s.info["cut_bottom_plane"] for s in subs} == {term["plane_type"]}


def test_bulk_matching_in_plane_supercell(ceo2_111_slab):
    from taskerslabgen import cutslab

    subs = cutslab(ceo2_111_slab * (2, 2, 1), Q_CEO2, bulk_atoms=CEO2)
    assert [s.get_chemical_formula() for s in subs] == ["Ce16O32", "Ce32O64", "Ce48O96"]


def test_bulk_matching_with_tasker3_reconstruction():
    from taskerslabgen import cutslab, generate_slabs_for_miller

    res = generate_slabs_for_miller(
        CEO2, Q_CEO2, (0, 0, 1), [4], prefer_plane="O", bond_distances=BOND_DISTS_CEO2
    )
    term = next(iter(res[(0, 0, 1)].values()))
    subs = cutslab(term["atoms"][0], Q_CEO2, reconstruction=term["reconstruction"], bulk_atoms=CEO2)
    assert [s.get_chemical_formula() for s in subs] == [f"Ce{2 * m}O{4 * m}" for m in range(1, 9)]
    assert {s.info["cut_bottom_plane"] for s in subs} == {"O4-recon"}


def test_bulk_matching_needs_miller(ceo2_111_slab):
    from taskerslabgen import cutslab

    slab = ceo2_111_slab.copy()
    slab.info.pop("miller", None)
    with pytest.raises(ValueError, match="Miller"):
        cutslab(slab, Q_CEO2, bulk_atoms=CEO2)


def test_primed_label_matching():
    from taskerslabgen import plane_name_base, plane_name_matches

    assert plane_name_base("O4'") == "O4"
    assert plane_name_base("IrO2-a'") == "IrO2"
    assert plane_name_matches("O4", "O4'")
    assert plane_name_matches("IrO2-a", "IrO2-a'")
    assert plane_name_matches("IrO2", "IrO2-b'")
    assert not plane_name_matches("O4'", "O4")


def test_no_ase_future_warnings():
    from taskerslabgen import generate_slabs_for_miller

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        generate_slabs_for_miller(CEO2, Q_CEO2, (1, 1, 0), [2])
        generate_slabs_for_miller(
            CEO2, Q_CEO2, (0, 0, 1), [2], prefer_plane="O", bond_distances=BOND_DISTS_CEO2
        )
