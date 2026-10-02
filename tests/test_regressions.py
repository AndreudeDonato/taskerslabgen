"""
Regression tests for the issues found in the reviews of 0.3.1 and of the fix branch.

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
ALBITE = read((BULK_DIR / "NaAlSi3O8_albite.cif").as_posix())

Q_CEO2 = {"Ce": 4.0, "O": -2.0}
Q_IRO2 = {"Ir": 4.0, "O": -2.0}
Q_ZNO = {"Zn": 2.0, "O": -2.0}
Q_GAN = {"Ga": 3.0, "N": -3.0}
Q_MGO = {"Mg": 2.0, "O": -2.0}
Q_ALBITE = {"Na": 1.0, "Al": 3.0, "Si": 4.0, "O": -2.0}
BOND_DISTS_CEO2 = {"Ce-Ce": None, "O-O": None, "Ce-O": 2.35}


# ------------------------------------------------------------------
# helpers (independent of the library internals)
# ------------------------------------------------------------------
def _tasker_type(atoms, charges, hkl, plane_tol=None):
    from taskerslabgen.advanced import (
        build_surface,
        compute_projection,
        compute_reduced_counts,
        enumerate_cut_pairs,
        identify_planes,
        select_best_sequence,
    )

    from taskerslabgen.core import _charge_scale, _surface_area

    surf = build_surface(atoms, hkl, layers=1)
    atoms_z, L = compute_projection(atoms, surf, charges, hkl)
    planes = identify_planes(atoms_z, L, plane_tol=plane_tol)
    best = select_best_sequence(
        enumerate_cut_pairs(planes, L, compute_reduced_counts(atoms_z),
                            area=_surface_area(surf.cell), charge_scale=_charge_scale(atoms_z[:, 2]))
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


def _assert_valid_slab(slab, charges, reduced, max_gap=None, polarity=None):
    """Stoichiometric, neutral, non-polar (|dipole| < 1e-4 e*A, or a
    normalised dipole per area of at most *polarity*) and without gaps above
    *max_gap*."""
    counts = Counter(int(z) for z in slab.numbers)
    ks = {counts.get(z, 0) / r for z, r in reduced.items()}
    assert len(ks) == 1 and next(iter(ks)) >= 1 and float(next(iter(ks))).is_integer(), (
        f"non-stoichiometric slab {slab.get_chemical_formula()}"
    )
    q = np.array([charges[s] for s in slab.get_chemical_symbols()])
    assert abs(q.sum()) < 1e-6, f"charged slab {slab.get_chemical_formula()} Q={q.sum():+.3f}"
    z = slab.positions[:, 2]
    mu = float(np.sum(q * (z - z.mean())))
    if polarity is None:
        assert abs(mu) < 1e-4, f"polar slab {slab.get_chemical_formula()} mu={mu:+.4f}"
    else:
        from taskerslabgen import dipole_per_area

        p = dipole_per_area(q, slab.positions, slab.cell)
        assert p <= polarity + 1e-12, (
            f"polar slab {slab.get_chemical_formula()} mu={mu:+.4f} (polarity {p:.2e} /A)"
        )
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

    subs = cutslab(ceo2_111_slab, Q_CEO2, cut_at="all", cuts="top")
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
    from taskerslabgen import PolarSurfaceError, generate_slabs_for_miller

    with pytest.raises(PolarSurfaceError, match="(?i)dipole") as err:
        generate_slabs_for_miller(ZNO * (2, 2, 1), Q_ZNO, (0, 0, 1), [2])
    assert err.value.min_dipole > 0.02  # genuinely polar (dipole_tol default 1e-3)


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
    from taskerslabgen.advanced import build_adjacency_matrix, build_surface

    surf = build_surface(atoms, hkl, layers=1)
    adj = build_adjacency_matrix(surf, bulk_atoms=atoms, miller=hkl)
    np.testing.assert_array_equal(np.asarray(adj, dtype=int), _reference_bond_counts(atoms, hkl))


def test_adjacency_with_bulk_requires_miller():
    from taskerslabgen.advanced import build_adjacency_matrix, build_surface

    surf = build_surface(CEO2, (1, 1, 1), layers=1)
    with pytest.raises(ValueError, match="miller"):
        build_adjacency_matrix(surf, bulk_atoms=CEO2)


# ------------------------------------------------------------------
# C6 / 0.5: a plane's label is its arrangement and its relative phase
# ------------------------------------------------------------------
def test_labels_give_arrangement_and_relative_phase(ceo2_111_slab):
    """Copies one lattice repeat apart share a label; the same arrangement
    in another phase (shifted or rotated) gets a prime."""
    from taskerslabgen.advanced import identify_planes
    from taskerslabgen.core import _charges_to_list, _slab_plane_names

    slab = ceo2_111_slab
    atoms_z = np.column_stack([
        slab.numbers, slab.positions[:, 2], _charges_to_list(slab, Q_CEO2)
    ])
    planes = sorted(identify_planes(atoms_z, slab.cell[2, 2]), key=lambda p: p["z_center"])
    names, repeat = _slab_plane_names(slab, planes)
    assert repeat is not None and repeat[0] == 3
    assert names == ["O4", "Ce4", "O4'"] * 3, names
    # Without genslab's hint the slab is labelled from its own interior.
    bare = slab.copy()
    bare.info.pop("stacking_labels", None)
    names_bare, _ = _slab_plane_names(bare, planes)
    assert [n.rstrip("'") for n in names_bare] == ["O4", "Ce4", "O4"] * 3


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
        top = term["top_plane_type"]
        # cutslab gives the slab's surface planes the labels genslab reported ...
        by_termination = cutslab(slab, charges, reconstruction=recon)
        assert {s.info["cut_bottom_plane"] for s in by_termination} == {label}
        assert {s.info["cut_top_plane"] for s in by_termination} == {top}
        # ... so genslab's labels can be passed straight to cut_at.
        by_label = cutslab(slab, charges, cut_at=[label, top], reconstruction=recon)
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
        # The label must stay attached to the same plane geometry, not only
        # the set of labels (letters of variants could swap).
        seen.add(frozenset(
            (info["plane_type"], _bottom_plane_descriptor(info["atoms"][0]))
            for info in res[hkl].values()
        ))
    assert len(seen) == 1, f"origin-dependent labels: {seen}"


def _plane_descriptor(atoms, indices):
    """Translation-invariant geometry of a plane: species-ordered in-plane
    difference vectors (fractional, 2 decimals)."""
    frac = atoms.get_scaled_positions()
    out = []
    for i in indices:
        for j in indices:
            if i != j and atoms.numbers[i] <= atoms.numbers[j]:
                d = np.round((frac[j, :2] - frac[i, :2]) % 1.0, 2) % 1.0
                out.append((int(atoms.numbers[i]), int(atoms.numbers[j]), *map(float, d)))
    return tuple(sorted(out))


def _bottom_plane_descriptor(slab):
    from taskerslabgen.advanced import identify_planes

    atoms_z = np.column_stack([slab.numbers, slab.positions[:, 2], np.zeros(len(slab))])
    planes = sorted(identify_planes(atoms_z, slab.cell[2, 2]), key=lambda p: p["z_center"])
    return _plane_descriptor(slab, planes[0]["indices"])


@pytest.mark.parametrize(
    "atoms, charges, hkl",
    [(ALBITE, Q_ALBITE, (0, 0, 1)), (ALBITE, Q_ALBITE, (1, 2, 1)), (IRO2, Q_IRO2, (0, 0, 1))],
    ids=["albite001", "albite121", "IrO2001"],
)
def test_every_plane_label_follows_its_geometry(atoms, charges, hkl):
    """L1/SW4: planes straddling the cell boundary of ase.build.surface had a
    sheared geometry, so variant letters changed with the bulk origin."""
    from taskerslabgen.advanced import (
        assign_plane_names,
        build_surface,
        compute_projection,
        identify_planes,
    )

    def labels(bulk_atoms):
        surf = build_surface(bulk_atoms, hkl, layers=1)
        atoms_z, L = compute_projection(bulk_atoms, surf, charges, hkl)
        planes = sorted(identify_planes(atoms_z, L), key=lambda p: p["z_center"])
        names, _ = assign_plane_names(planes, atoms=surf)
        return frozenset((name, _plane_descriptor(surf, p["indices"]))
                         for name, p in zip(names, planes))

    rng = np.random.default_rng(5)
    shifts = [np.zeros(3)] + [rng.random(3) for _ in range(5)]
    if hkl == (0, 0, 1):
        # Put the cell boundary just below each atom in turn: its plane then
        # straddles z = 0 in ase.build.surface.
        shifts += [np.array([0.0, 0.0, 1e-3 - f]) for f in atoms.get_scaled_positions()[:, 2]]
    seen = {labels(_shifted(atoms, shift)) for shift in shifts}
    assert len(seen) == 1, "plane labels depend on the bulk origin"


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


def test_dipole_tol_is_per_area_of_the_thickest_slab():
    """The polarity is per surface area; n stacked repeat units have n times
    the dipole of one, so the Tasker type is decided for the thickest slab."""
    from taskerslabgen.advanced import select_best_sequence

    def seq(p):
        return {"is_neutral": True, "is_stoich": True, "is_full_period": True,
                "net_dipole": p, "stoich_k": 1, "dipole_per_area": p, "bottom_cut": 0}

    assert select_best_sequence([seq(4e-4)], dipole_tol=1e-3, n_units=2)["is_tasker_ii"]
    assert not select_best_sequence([seq(4e-4)], dipole_tol=1e-3, n_units=3)["is_tasker_ii"]


def test_polarity_ignores_the_scale_of_the_charges(ceo2_111_slab):
    """Formal, relative or computed charges in proportion give one polarity."""
    from taskerslabgen import dipole_per_area

    relaxed = ceo2_111_slab.copy()
    relaxed.positions[relaxed.positions[:, 2] > relaxed.positions[:, 2].max() - 1.0, 2] -= 0.1
    formal = np.array([Q_CEO2[s] for s in relaxed.get_chemical_symbols()])
    values = {dipole_per_area(scale * formal, relaxed.positions, relaxed.cell)
              for scale in (1.0, 0.5, 0.31)}
    assert max(values) - min(values) < 1e-12 and max(values) > 1e-3


def test_relaxed_sub_slabs_judged_alike_at_every_thickness(ceo2_111_slab):
    """A sub-slab keeps one relaxed surface: its dipole per area (~0.007 /A
    here) does not depend on the thickness, so the verdict does not either."""
    from taskerslabgen import cutslab

    relaxed = ceo2_111_slab.copy()
    z = relaxed.positions[:, 2]
    relaxed.positions[z < z.min() + 1.0, 2] += 0.10   # outer planes relax inward
    relaxed.positions[z > z.max() - 1.0, 2] -= 0.10
    relaxed.positions[:, 2] += np.random.default_rng(0).normal(0.0, 0.01, len(relaxed))
    subs = cutslab(relaxed, Q_CEO2, dipole_tol=0.05)
    assert [s.get_chemical_formula() for s in subs] == ["Ce4O8", "Ce8O16", "Ce12O24"]
    strict = cutslab(relaxed, Q_CEO2, dipole_tol=0.005)
    assert [s.get_chemical_formula() for s in strict] == ["Ce12O24"]


# ------------------------------------------------------------------
# N3: best Tasker I/II termination = fewest broken bonds, origin-independent
# ------------------------------------------------------------------
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
    subs = cutslab(rumpled, Q_CEO2, bulk_atoms=CEO2, dipole_tol=0.05)
    assert [s.get_chemical_formula() for s in subs] == ["Ce4O8", "Ce8O16", "Ce12O24"]
    assert {(s.info["cut_bottom_plane"], s.info["cut_top_plane"]) for s in subs} == {("O4", "O4'")}


def test_deformed_surface_plane_gets_tilde_label(ceo2_111_slab):
    from taskerslabgen import cutslab

    shifted = _relax_surfaces(ceo2_111_slab, "shifted")
    subs = cutslab(shifted, Q_CEO2, bulk_atoms=CEO2)
    assert [s.get_chemical_formula() for s in subs] == ["Ce4O8", "Ce8O16", "Ce12O24"]
    surfaces = [(s.info["cut_bottom_plane"], s.info["cut_top_plane"]) for s in subs]
    assert surfaces == [("O4~", "O4'"), ("O4~", "O4'"), ("O4~", "O4'~")]


def test_cut_dipoles_use_relaxed_positions(ceo2_111_slab):
    """Cuts keep one surface relaxed outwards by 0.2 A: 0.012 /A at every
    thickness, against ~0 for the symmetric input slab."""
    from taskerslabgen import cutslab

    relaxed = _relax_surfaces(ceo2_111_slab, "outward")
    subs = cutslab(relaxed, Q_CEO2, bulk_atoms=CEO2, dipole_tol=0.01)
    assert [s.get_chemical_formula() for s in subs] == ["Ce12O24"]
    subs = cutslab(relaxed, Q_CEO2, bulk_atoms=CEO2, dipole_tol=0.05)
    assert [s.get_chemical_formula() for s in subs] == ["Ce4O8", "Ce8O16", "Ce12O24"]


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


def test_label_grammar_and_selection():
    from taskerslabgen import plane_name_base, plane_name_matches
    from taskerslabgen.core import _parse_plane_name

    assert _parse_plane_name("IrO2-a''-recon~") == ("IrO2", "a", 2, True, True)
    assert _parse_plane_name("O'5") == ("O", None, 5, False, False)
    assert plane_name_base("O4~") == "O4"
    assert plane_name_base("IrO2-a'") == "IrO2"
    # relative (default): only that plane; deformed and reconstructed copies count
    assert plane_name_matches("O4", "O4~")
    assert plane_name_matches("O4'", "O4'-recon")
    assert not plane_name_matches("O4", "O4'")
    assert not plane_name_matches("O4~", "O4")
    assert not plane_name_matches("IrO2", "IrO2-a")
    # shape: the arrangement in any phase
    assert plane_name_matches("O4", "O4''", "shape")
    assert plane_name_matches("IrO2", "IrO2-b'", "shape")
    assert not plane_name_matches("IrO2-a", "IrO2-b", "shape")
    with pytest.raises(ValueError, match="selection"):
        plane_name_matches("O4", "O4", "loose")


# ------------------------------------------------------------------
# Step 5 (M3): default distribution score is the Coulomb energy
# ------------------------------------------------------------------
def test_default_pattern_spreads_like_charges():
    """Half-occupied O plane of CeO2(001): checkerboard, not rows, by default."""
    from taskerslabgen import generate_slabs_for_miller

    res = generate_slabs_for_miller(CEO2, Q_CEO2, (0, 0, 1), [2], prefer_plane="O")
    deleted = next(iter(res[(0, 0, 1)].values()))["reconstruction"]["delete_info"]
    assert len(deleted) == 2
    (_, x1, y1), (_, x2, y2) = deleted
    d = np.abs((np.array([x1 - x2, y1 - y2]) + 0.5) % 1.0 - 0.5)
    np.testing.assert_allclose(d, [0.5, 0.5], atol=1e-6)


def test_broken_bonds_reported_by_pair():
    from taskerslabgen import generate_slabs_for_miller

    q = {"Sr": 2.0, "Ti": 4.0, "O": -2.0}
    cand = next(iter(generate_slabs_for_miller(_srtio3(), q, (1, 0, 0), [2])[(1, 0, 0)].values()))["candidate"]
    assert sum(cand["broken_bonds_by_pair"].values()) == cand["broken_bonds"]
    assert {"O-Sr", "O-Ti"} <= set(cand["broken_bonds_by_pair"])
    # Cation-cation contacts can be excluded explicitly.
    no_cc = {"Sr-Sr": None, "Sr-Ti": None, "Ti-Ti": None}
    cand = next(iter(generate_slabs_for_miller(
        _srtio3(), q, (1, 0, 0), [2], bond_distances=no_cc)[(1, 0, 0)].values()))["candidate"]
    assert cand["broken_bonds_by_pair"] == {"O-Sr": 4, "O-Ti": 1}


# ------------------------------------------------------------------
# M9: cutslab on a slab that straddles the cell boundary
# ------------------------------------------------------------------
def test_cutslab_handles_slab_wrapped_across_cell_boundary(ceo2_111_slab):
    from taskerslabgen import cutslab

    centred = ceo2_111_slab.copy()
    centred.positions[:, 2] -= centred.positions[:, 2].mean()
    centred.wrap()
    subs = cutslab(centred, Q_CEO2)
    assert [s.get_chemical_formula() for s in subs] == ["Ce4O8", "Ce8O16", "Ce12O24"]


def test_no_ase_future_warnings():
    from taskerslabgen import generate_slabs_for_miller

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        generate_slabs_for_miller(CEO2, Q_CEO2, (1, 1, 0), [2])
        generate_slabs_for_miller(
            CEO2, Q_CEO2, (0, 0, 1), [2], prefer_plane="O", bond_distances=BOND_DISTS_CEO2
        )


# ------------------------------------------------------------------
# Second review pass: validation, charges, cut positions
# ------------------------------------------------------------------
ANATASE = read((BULK_DIR / "TiO2_anatase.cif").as_posix())


def test_anatase_101_slabs_are_neutral_and_stoichiometric():
    """0.3.1 returned Ti14O26 (charge +4) slabs for anatase (101)."""
    from taskerslabgen import generate_slabs_for_miller

    q = {"Ti": 4.0, "O": -2.0}
    res = generate_slabs_for_miller(ANATASE, q, (1, 0, 1), [3, 4, 5])
    info = next(iter(res[(1, 0, 1)].values()))
    assert info["tasker_type"] == "I/II"
    counts = [Counter(s.get_chemical_symbols()) for s in info["atoms"]]
    assert [(c["Ti"], c["O"]) for c in counts] == [(12, 24), (16, 32), (20, 40)]
    for slab in info["atoms"]:
        _assert_valid_slab(slab, q, _reduced(ANATASE))


def test_tasker3_slabs_of_noisy_bulk_are_accepted():
    """G1: a rumpled reconstructed plane widens a gap; that is not a glued slab."""
    from taskerslabgen import generate_slabs_for_miller

    for seed in range(5):
        noisy = CEO2.copy()
        noisy.positions += np.random.default_rng(seed).normal(0.0, 1e-5, noisy.positions.shape)
        res = generate_slabs_for_miller(noisy, Q_CEO2, (0, 0, 1), [1, 2])
        assert res[(0, 0, 1)]


def test_cutslab_on_slab_with_fixed_atoms(ceo2_111_slab):
    """B2: FixAtoms must not hold atoms back when the vacuum is moved."""
    from ase.constraints import FixAtoms
    from taskerslabgen import cutslab

    slab = ceo2_111_slab.copy()
    z = slab.positions[:, 2]
    slab.set_constraint(FixAtoms(mask=z < z.min() + 2.0))
    subs = cutslab(slab, Q_CEO2)
    assert [s.get_chemical_formula() for s in subs] == ["Ce4O8", "Ce8O16", "Ce12O24"]


def test_cutslab_without_vacuum_keeps_parent_coordinates(ceo2_111_slab):
    """B4: a slab already inside its cell is not shifted."""
    from taskerslabgen import cutslab

    full = cutslab(ceo2_111_slab, Q_CEO2, vacuum=0)[-1]
    np.testing.assert_allclose(full.positions, ceo2_111_slab.positions)


def test_charge_tol_is_per_formula_unit():
    """G2: a 1e-4 e charge excess per Ce is noise in any cell size."""
    from taskerslabgen import generate_slabs_for_miller

    supercell = read((BULK_DIR / "CeO2_fluorite_supercell2x2x2.cif").as_posix())
    res = generate_slabs_for_miller(supercell, {"Ce": 4.0001, "O": -2.0}, (1, 1, 1), [1])
    assert res[(1, 1, 1)]


@pytest.mark.parametrize("hkl, plane_tol", [((-1, 1, -1), 0.2), ((-1, 0, 1), 0.3)])
def test_cut_does_not_slice_a_thick_plane(hkl, plane_tol):
    """SW1: cuts go in the gap between plane extents, not between plane centres."""
    from taskerslabgen import generate_slabs_for_miller

    res = generate_slabs_for_miller(ALBITE, Q_ALBITE, hkl, [1, 2], plane_tol=plane_tol,
                                    candidates="all")
    for info in res[hkl].values():
        for slab in info["atoms"]:
            _assert_valid_slab(slab, Q_ALBITE, _reduced(ALBITE))


def test_unknown_termination_id_raises():
    """SW3: prefer_plane=<missing id> used to return an empty result."""
    from taskerslabgen import generate_slabs_for_miller

    with pytest.raises(ValueError, match="available IDs"):
        generate_slabs_for_miller(CEO2, Q_CEO2, (1, 1, 1), [1], prefer_plane=99)


def test_plot_creates_output_directory(tmp_path):
    """SW6"""
    from taskerslabgen import generate_slabs_for_miller

    out = tmp_path / "new" / "plots"
    generate_slabs_for_miller(CEO2, Q_CEO2, (1, 1, 1), [1], plot=True, plot_out_dir=str(out))
    assert list(out.glob("*.png"))


def test_hirshfeld_parser_keeps_last_analysis(tmp_path):
    """SW7: a relaxation prints one Hirshfeld block per geometry."""
    from taskerslabgen import parse_hirshfeld_fhi_aims

    block = "  Performing Hirshfeld analysis of fragment charges and moments.\n"
    out = tmp_path / "aims.out"
    out.write_text(
        block + "  |   Hirshfeld charge        :      0.10\n"
        "  |   Hirshfeld charge        :     -0.10\n"
        + block + "  |   Hirshfeld charge        :      0.30\n"
        "  |   Hirshfeld charge        :     -0.30\n"
    )
    assert parse_hirshfeld_fhi_aims(out) == [0.30, -0.30]


def test_bulk_matching_oblique_supercell_with_cell_noise(ceo2_111_slab):
    """L4: in-plane tiling must not depend on rounding of fractional positions."""
    from ase.build import make_supercell
    from taskerslabgen import cutslab

    sc = make_supercell(ceo2_111_slab, [[2, 1, 0], [-1, 1, 0], [0, 0, 1]])
    sc.set_cell(sc.cell * (1 + 1e-6), scale_atoms=True)
    subs = cutslab(sc, Q_CEO2, bulk_atoms=CEO2, miller=(1, 1, 1))
    assert [s.get_chemical_formula() for s in subs] == ["Ce12O24", "Ce24O48", "Ce36O72"]


def test_charges_read_from_atoms(ceo2_111_slab):
    """charges=None uses the charges ASE stores on the structure."""
    from ase.calculators.singlepoint import SinglePointCalculator
    from taskerslabgen import cutslab, generate_slabs_for_miller

    bulk_q = CEO2.copy()
    bulk_q.set_initial_charges([Q_CEO2[s] for s in bulk_q.get_chemical_symbols()])
    res = generate_slabs_for_miller(bulk_q, None, (1, 1, 1), [2])
    assert next(iter(res[(1, 1, 1)].values()))["atoms"][0].get_chemical_formula() == "Ce8O16"

    slab = ceo2_111_slab.copy()
    noise = np.random.default_rng(0).normal(0.0, 1e-4, len(slab))
    slab.calc = SinglePointCalculator(
        slab, charges=np.array([Q_CEO2[s] for s in slab.get_chemical_symbols()]) + noise
    )
    subs = cutslab(slab, None)
    assert [s.get_chemical_formula() for s in subs] == ["Ce4O8", "Ce8O16", "Ce12O24"]
    with pytest.raises(ValueError, match="set_initial_charges"):
        cutslab(ceo2_111_slab, None)


def test_oriented_bulk_layers_are_consistent_copies():
    """L1: copy m of every atom sits m*L above copy 0, and no plane straddles z=0."""
    from taskerslabgen.advanced import build_surface

    hkl = (0, 0, 1)
    for f in ALBITE.get_scaled_positions()[::4, 2]:
        bulk_atoms = _shifted(ALBITE, [0.0, 0.0, 1e-3 - f])  # boundary just below an atom
        one = build_surface(bulk_atoms, hkl, layers=1)
        three = build_surface(bulk_atoms, hkl, layers=3)
        L = one.cell[2, 2]
        z1 = one.positions[:, 2]
        assert z1.min() > 0.05 and z1.max() < L - 0.05
        expected = np.sort(np.concatenate([z1 + m * L for m in range(3)]))
        np.testing.assert_allclose(np.sort(three.positions[:, 2]), expected, atol=1e-8)


def test_variant_letters_stable_under_small_displacements():
    """L2: 0.005 A of in-plane noise swapped IrO2-a / IrO2-b."""
    from taskerslabgen import cutslab, generate_slabs_for_miller

    slab = next(iter(generate_slabs_for_miller(IRO2, Q_IRO2, (0, 0, 1), [3])[(0, 0, 1)].values()))["atoms"][0]
    ref = [s.info["cut_bottom_plane"] for s in cutslab(slab, Q_IRO2, cut_at="all", cuts="all")]
    for seed in range(10):
        noisy = slab.copy()
        noisy.positions[:, :2] += np.random.default_rng(seed).normal(0.0, 0.005, (len(noisy), 2))
        got = [s.info["cut_bottom_plane"] for s in cutslab(noisy, Q_IRO2, cut_at="all", cuts="all")]
        assert got == ref


# ------------------------------------------------------------------
# Second review pass: exact Tasker III scoring
# ------------------------------------------------------------------
def _corundum():
    from ase.spacegroup import crystal

    return crystal(["Al", "O"], [(0, 0, 0.3523), (0.3064, 0, 0.25)], spacegroup=167,
                   cellpar=[4.759, 4.759, 12.99, 90, 90, 120])


@pytest.mark.parametrize(
    "atoms, charges, hkl, kwargs",
    [(_corundum(), {"Al": 3.0, "O": -2.0}, (1, 1, 1), {}),
     (ALBITE, Q_ALBITE, (0, 1, 1), {"plane_tol": 0.2})],
    ids=["corundum111", "albite011"],
)
def test_tasker3_dipole_counts_the_deleted_atoms(atoms, charges, hkl, kwargs):
    """T1: on rumpled planes the dipole depends on which atoms are deleted;
    plane centres scored every pattern as zero and the built slab was polar."""
    from taskerslabgen import generate_slabs_for_miller

    res = generate_slabs_for_miller(atoms, charges, hkl, [1, 2], candidates="all", **kwargs)
    assert res[hkl]
    for info in res[hkl].values():
        assert info["tasker_type"] == "III"
        for slab in info["atoms"]:
            # Rumpled surface planes leave a small dipole; within dipole_tol.
            _assert_valid_slab(slab, charges, _reduced(atoms), polarity=1e-3)


@pytest.mark.parametrize(
    "atoms, charges, hkl",
    [(CEO2, Q_CEO2, (0, 0, 1)), (MGO, Q_MGO, (1, 1, 1))],
    ids=["CeO2001", "MgO111"],
)
def test_tasker3_best_independent_of_bulk_origin(atoms, charges, hkl):
    """T2: zero dipoles differing by 1e-15 used to decide the best pattern."""
    from taskerslabgen import generate_slabs_for_miller

    rng = np.random.default_rng(2)
    best = set()
    for shift in [np.zeros(3)] + [rng.random(3) for _ in range(8)]:
        info = next(iter(generate_slabs_for_miller(_shifted(atoms, shift), charges, hkl, [2])[hkl].values()))
        best.add((info["plane_type"], info["candidate"]["bond_score"]))
    assert len(best) == 1, f"origin-dependent Tasker III choice: {best}"


def _coordination(atoms, lo=0.85, hi=1.15):
    radii = covalent_radii[atoms.numbers]
    i_idx, j_idx, d = neighbor_list("ijd", atoms, hi * 2.0 * float(radii.max()))
    ref = radii[i_idx] + radii[j_idx]
    ok = (d >= lo * ref) & (d <= hi * ref)
    return np.bincount(i_idx[ok], minlength=len(atoms))


@pytest.mark.parametrize(
    "atoms, charges, hkl", [(MGO, Q_MGO, (1, 1, 1)), (CEO2, Q_CEO2, (0, 0, 1))],
    ids=["MgO111", "CeO2001"],
)
def test_tasker3_bond_score_counts_dangling_bonds(atoms, charges, hkl):
    """T3: the bond score is the number of bulk bonds the slab's atoms lose."""
    from taskerslabgen import generate_slabs_for_miller

    bulk_coord = {int(z): c for z, c in zip(atoms.numbers, _coordination(atoms))}
    res = generate_slabs_for_miller(atoms, charges, hkl, [4], candidates="all")
    for info in res[hkl].values():
        slab = info["atoms"][0].copy()
        slab.pbc = (True, True, False)
        expected = np.array([bulk_coord[int(z)] for z in slab.numbers])
        deficit = int(np.sum(expected - _coordination(slab)))
        assert info["candidate"]["bond_score"] == deficit


def test_surface_supercell_keeps_the_facet():
    """T4: bulk * (2, 2, 1) changes the facet; surface_supercell does not."""
    from taskerslabgen import generate_slabs_for_miller
    from taskerslabgen.advanced import build_surface

    prim = bulk("MgO", "rocksalt", a=4.21)
    with pytest.raises(ValueError, match="surface_supercell"):
        generate_slabs_for_miller(prim, Q_MGO, (1, 1, 1), [2])
    res = generate_slabs_for_miller(prim, Q_MGO, (1, 1, 1), [2], surface_supercell=(2, 2))
    slab = next(iter(res[(1, 1, 1)].values()))["atoms"][0]
    one = build_surface(prim, (1, 1, 1), layers=1)
    np.testing.assert_allclose(slab.cell[:2], 2 * one.cell[:2], atol=1e-8)
    assert slab.info["miller"] == (1, 1, 1)
    _assert_valid_slab(slab, Q_MGO, _reduced(prim))


def test_tasker3_prefer_plane_accepts_recon_labels():
    """SW2: genslab's own plane_type ("O4-recon") must work as prefer_plane."""
    from taskerslabgen import generate_slabs_for_miller, reconstruct_tasker_iii

    res = generate_slabs_for_miller(CEO2, Q_CEO2, (0, 0, 1), [2], prefer_plane="O4-recon")
    assert [t["plane_type"] for t in res[(0, 0, 1)].values()] == ["O4-recon"]
    out = reconstruct_tasker_iii(CEO2, Q_CEO2, (0, 0, 1), [2], "CeO2", prefer_plane="Ce2-recon")
    assert out["best_candidate"]["recon_label"] == "Ce2-recon"


def test_tasker3_errors_name_the_failed_condition():
    """T5: a charge failure was reported as a dipole failure."""
    from taskerslabgen.tasker3 import _select_tasker3_candidates

    charged = {"is_neutral": False, "charge_per_fu": 0.5, "dipole_per_area": 0.0}
    with pytest.raises(ValueError, match="charge-neutral"):
        _select_tasker3_candidates([charged], (0, 0, 1), 1e-3, 1e-3)
    polar = {"is_neutral": True, "charge_per_fu": 0.0, "dipole_per_area": 0.1}
    with pytest.raises(ValueError, match="dipole"):
        _select_tasker3_candidates([polar], (0, 0, 1), 1e-3, 1e-3)


# ------------------------------------------------------------------
# Second review pass: re-applying Tasker III reconstructions in cutslab
# ------------------------------------------------------------------
def _same_structure(a, b, tol=1e-3):
    """Same atoms up to a rigid shift along z and in-plane lattice vectors."""
    from scipy.optimize import linear_sum_assignment

    if a.get_chemical_formula() != b.get_chemical_formula():
        return False
    pa, pb = a.positions.copy(), b.positions.copy()
    pa[:, 2] -= pa[:, 2].min()
    pb[:, 2] -= pb[:, 2].min()
    cell = a.cell[:2, :2]
    for z in set(a.numbers):
        ia, ib = np.flatnonzero(a.numbers == z), np.flatnonzero(b.numbers == z)
        d = pa[ia][:, None, :] - pb[ib][None, :, :]
        f = d[..., :2] @ np.linalg.inv(cell)
        d[..., :2] = (f - np.round(f)) @ cell
        dist = np.linalg.norm(d, axis=-1)
        rows, cols = linear_sum_assignment(dist)
        if dist[rows, cols].max() > tol:
            return False
    return True


@pytest.mark.parametrize(
    "atoms, charges, hkl",
    [(MGO, Q_MGO, (1, 1, 1)), (_srtio3(), {"Sr": 2.0, "Ti": 4.0, "O": -2.0}, (1, 1, 0)),
     (CEO2, Q_CEO2, (0, 0, 1)), (_corundum(), {"Al": 3.0, "O": -2.0}, (1, 1, 1))],
    ids=["MgO111", "SrTiO3110", "CeO2001", "corundum111"],
)
def test_cutslab_reconstruction_reproduces_genslab(atoms, charges, hkl):
    """R4: planes that map onto themselves under a non-lattice shift made the
    re-applied pattern depend on atom order; sub-slabs must equal genslab's."""
    from taskerslabgen import cutslab, generate_slabs_for_miller

    res = generate_slabs_for_miller(atoms, charges, hkl, [1, 2, 3], candidates="all")
    for term in list(res[hkl].values())[:4]:
        by_size = {len(s): s for s in term["atoms"]}
        thick = term["atoms"][-1]
        order = np.random.default_rng(0).permutation(len(thick))
        for slab in (thick, thick[order]):
            subs = cutslab(slab, charges, reconstruction=term["reconstruction"])
            assert {len(s) for s in subs} >= set(by_size)
            for s in subs:
                if len(s) in by_size:
                    assert _same_structure(s, by_size[len(s)])


def test_cutslab_reconstruction_on_supercell_and_after_json():
    """R1: an in-plane supercell or a JSON round trip silently lost the series."""
    import json
    from taskerslabgen import cutslab, generate_slabs_for_miller

    res = generate_slabs_for_miller(CEO2, Q_CEO2, (0, 0, 1), [4], prefer_plane="O",
                                    bond_distances=BOND_DISTS_CEO2)
    term = next(iter(res[(0, 0, 1)].values()))
    slab, recon = term["atoms"][0], term["reconstruction"]
    subs = cutslab(slab * (2, 2, 1), Q_CEO2, reconstruction=recon)
    assert [s.get_chemical_formula() for s in subs] == [f"Ce{8 * m}O{16 * m}" for m in range(1, 9)]
    subs = cutslab(slab, Q_CEO2, reconstruction=json.loads(json.dumps(recon)))
    assert [s.get_chemical_formula() for s in subs] == [f"Ce{2 * m}O{4 * m}" for m in range(1, 9)]
    with pytest.raises(ValueError, match="same bulk and Miller index"):
        cutslab(next(iter(generate_slabs_for_miller(CEO2, Q_CEO2, (1, 1, 1), [2])[(1, 1, 1)].values()))["atoms"][0],
                Q_CEO2, reconstruction=recon)


def test_cutslab_cut_at_is_not_overridden_by_reconstruction():
    """R2: reconstructed copies were cut even when cut_at named another plane."""
    from taskerslabgen import cutslab, generate_slabs_for_miller

    res = generate_slabs_for_miller(CEO2, Q_CEO2, (0, 0, 1), [4], prefer_plane="O",
                                    bond_distances=BOND_DISTS_CEO2)
    term = next(iter(res[(0, 0, 1)].values()))
    with pytest.raises(ValueError, match="No stoichiometric"):
        cutslab(term["atoms"][0], Q_CEO2, reconstruction=term["reconstruction"],
                cut_at="Ce2", cuts="all")


def test_top_plane_type_for_asymmetric_terminations():
    """R3: plane_type is the bottom plane only; cut_at needs both surfaces."""
    from taskerslabgen import cutslab, generate_slabs_for_miller

    res = generate_slabs_for_miller(ALBITE, Q_ALBITE, (0, 1, 0), [3], candidates="all")
    asymmetric = [t for t in res[(0, 1, 0)].values() if t["plane_type"] != t["top_plane_type"]]
    assert asymmetric
    for term in asymmetric:
        subs = cutslab(term["atoms"][0], Q_ALBITE, cut_at=[term["plane_type"], term["top_plane_type"]])
        assert len(subs) == 3


def test_cutslab_warns_when_surface_planes_never_occur_inside(recwarn):
    """SW5: a Tasker III slab without reconstruction= silently gave only itself."""
    from taskerslabgen import cutslab, generate_slabs_for_miller

    res = generate_slabs_for_miller(CEO2, Q_CEO2, (0, 0, 1), [3], prefer_plane="O",
                                    bond_distances=BOND_DISTS_CEO2)
    slab = next(iter(res[(0, 0, 1)].values()))["atoms"][0]
    with pytest.warns(UserWarning, match="reconstruction="):
        assert len(cutslab(slab, Q_CEO2)) == 1


# ------------------------------------------------------------------
# cutslab bulk matching with supercell bulks, strain and split planes
# (relaxed FHI-aims slabs of rutile/anatase/fluorite oxides failed)
# ------------------------------------------------------------------
def test_bulk_matching_with_supercell_bulk(ceo2_111_slab):
    """A relaxed bulk is often a supercell of the cell the slab was cut from."""
    from taskerslabgen import cutslab

    supercell = read((BULK_DIR / "CeO2_fluorite_supercell2x2x2.cif").as_posix())
    subs = cutslab(ceo2_111_slab, Q_CEO2, bulk_atoms=supercell)
    assert [s.get_chemical_formula() for s in subs] == ["Ce4O8", "Ce8O16", "Ce12O24"]
    assert {s.info["cut_bottom_plane"] for s in subs} == {"O4"}


def test_bulk_matching_tolerates_small_strain():
    """Slab built from a lattice 0.6% larger than bulk_atoms (another calculation)."""
    from taskerslabgen import cutslab, generate_slabs_for_miller

    strained = CEO2.copy()
    strained.set_cell(CEO2.cell * 1.006, scale_atoms=True)
    slab = next(iter(generate_slabs_for_miller(strained, Q_CEO2, (1, 1, 1), [6])[(1, 1, 1)].values()))["atoms"][0]
    subs = cutslab(slab, Q_CEO2, bulk_atoms=CEO2)
    assert [s.get_chemical_formula() for s in subs] == [f"Ce{4 * m}O{8 * m}" for m in range(1, 7)]


def test_bulk_matching_when_relaxation_splits_every_plane():
    """No slab plane has a bulk composition: register single atoms instead."""
    from taskerslabgen import cutslab, generate_slabs_for_miller

    slab = next(iter(generate_slabs_for_miller(IRO2, Q_IRO2, (0, 0, 1), [4])[(0, 0, 1)].values()))["atoms"][0]
    split = slab.copy()
    oxygens = np.flatnonzero(split.numbers == 8)
    # Within each layer, one O moves up and one down: every IrO2 plane
    # splits into O / Ir / O at plane_tol = 0.1 A, without a dipole.
    order = oxygens[np.lexsort((split.positions[oxygens, 0], np.round(split.positions[oxygens, 2], 1)))]
    split.positions[order[0::2], 2] += 0.12
    split.positions[order[1::2], 2] -= 0.12
    subs = cutslab(split, Q_IRO2, bulk_atoms=IRO2, dipole_tol=0.05)
    # The planes alternate IrO2 / IrO2' (rotated): keeping the input's top
    # plane takes every second thickness, any phase takes all of them.
    assert [len(s) for s in subs] == [6 * m for m in range(1, 5)]
    assert {s.info["cut_bottom_plane"] for s in subs} == {"IrO2'~"}
    loose = cutslab(split, Q_IRO2, bulk_atoms=IRO2, dipole_tol=0.05, selection="shape")
    assert [len(s) for s in loose] == [3 * m for m in range(1, 9)]


def test_cutslab_reconstruction_with_supercell_bulk():
    """The 2x2x2 CeO2 CIF stacks copies of the O plane every quarter of its
    period; whether they were whole repeat units apart depended on rounding.
    (2x2 in-plane: about 5 s of Tasker III enumeration.)"""
    from taskerslabgen import cutslab, generate_slabs_for_miller

    supercell = read((BULK_DIR / "CeO2_fluorite_supercell2x2x2.cif").as_posix())
    res = generate_slabs_for_miller(supercell, Q_CEO2, (0, 0, 1), [2], prefer_plane="O",
                                    bond_distances=BOND_DISTS_CEO2)
    term = next(iter(res[(0, 0, 1)].values()))
    subs = cutslab(term["atoms"][0], Q_CEO2, reconstruction=term["reconstruction"])
    assert [s.get_chemical_formula() for s in subs] == [f"Ce{8 * m}O{16 * m}" for m in range(1, 9)]
    for s in subs:
        _assert_valid_slab(s, Q_CEO2, _reduced(CEO2))


# ------------------------------------------------------------------
# C7 + M7: symmetry-equivalent Tasker III patterns are one termination
# ------------------------------------------------------------------
def test_symmetry_equivalent_patterns_are_one_termination():
    """CeO2 (001): 16 deletion patterns, 3 distinct terminations (O4
    checkerboard, O4 rows, half Ce plane); the two O4 planes of the
    conventional cell are related by an FCC translation."""
    from taskerslabgen import generate_slabs_for_miller

    res = generate_slabs_for_miller(CEO2, Q_CEO2, (0, 0, 1), [2], candidates="all")
    terms = res[(0, 0, 1)]
    assert sorted(t["plane_type"] for t in terms.values()) == ["Ce2-recon", "O4-recon", "O4-recon"]
    assert sum(t["candidate"]["multiplicity"] for t in terms.values()) == 16
    assert sorted(terms) == list(range(len(terms)))


def test_symmetry_reduction_keeps_the_scores():
    """Every distinct score of the full enumeration survives the reduction."""
    import taskerslabgen.tasker3 as t3
    from taskerslabgen.advanced import (
        assign_plane_names,
        build_surface,
        compute_projection,
        compute_reduced_counts,
        identify_planes,
    )

    def candidates(identity_only):
        surf = build_surface(MGO, (1, 1, 1), layers=1)
        atoms_z, L = compute_projection(MGO, surf, Q_MGO, (1, 1, 1))
        planes = sorted(identify_planes(atoms_z, L), key=lambda p: p["z_center"])
        names, _ = assign_plane_names(planes, atoms=surf)
        original = t3._stacking_symmetry
        if identity_only:
            t3._stacking_symmetry = lambda numbers, *a, **k: [np.arange(len(numbers))]
        try:
            return t3.find_tasker3_candidates(planes, atoms_z, compute_reduced_counts(atoms_z), None, L,
                                              surf_bulk=surf, plane_names=names, bulk_atoms=MGO,
                                              miller=(1, 1, 1))
        finally:
            t3._stacking_symmetry = original

    def scores(cands):
        return {(c["recon_label"], c["bond_score"], round(c["distribution_score"], 6)) for c in cands}

    full, reduced = candidates(True), candidates(False)
    assert len(full) == 12 and len(reduced) == 2
    assert scores(full) == scores(reduced)
    assert sum(c["multiplicity"] for c in reduced) == len(full)


def test_max_masks_guard():
    from taskerslabgen import generate_slabs_for_miller

    with pytest.raises(ValueError, match="max_masks"):
        generate_slabs_for_miller(CEO2, Q_CEO2, (0, 0, 1), [2], max_masks=10)


# ------------------------------------------------------------------
# S4: public API is the workflow; lower-level helpers in .advanced
# ------------------------------------------------------------------
def test_public_api_and_deprecated_names(ceo2_111_slab):
    import taskerslabgen
    from taskerslabgen import advanced

    assert {"generate_slabs_for_miller", "cutslab", "reconstruct_tasker_iii"} <= set(taskerslabgen.__all__)
    assert not set(taskerslabgen.__all__) & set(advanced.__all__)
    with pytest.warns(DeprecationWarning, match="taskerslabgen.advanced.build_surface"):
        assert taskerslabgen.build_surface is advanced.build_surface
    with pytest.raises(AttributeError, match="removed"):
        taskerslabgen.extract_termination
    with pytest.warns(DeprecationWarning, match="bond_threshold"):
        taskerslabgen.cutslab(ceo2_111_slab, Q_CEO2, bond_threshold=(0.8, 1.2))


# ------------------------------------------------------------------
# dipole_tol_max: opt-in fallback for slightly distorted bulks
# ------------------------------------------------------------------
def _distorted_iro2(shift=0.05):
    """IrO2 with one Ir moved along a: (110) becomes slightly polar
    (least polar 4-layer slab ~0.008 /A, normalised dipole per area)."""
    b = IRO2.copy()
    i = [a.index for a in b if a.symbol == "Ir"][0]
    b.positions[i] += [shift, 0.0, 0.0]
    return b


def test_dipole_tol_max_builds_slightly_polar_facet_with_warning():
    from taskerslabgen import PolarSurfaceError, cutslab, generate_slabs_for_miller

    b = _distorted_iro2()
    with pytest.raises(PolarSurfaceError) as err:
        generate_slabs_for_miller(b, Q_IRO2, (1, 1, 0), [4])
    needed = err.value.min_dipole
    assert 1e-3 < needed < 0.02
    # A cap below what the facet needs still raises.
    with pytest.raises(PolarSurfaceError):
        generate_slabs_for_miller(b, Q_IRO2, (1, 1, 0), [4], dipole_tol_max=0.9 * needed)

    with pytest.warns(UserWarning, match="dipole_tol"):
        result = generate_slabs_for_miller(b, Q_IRO2, (1, 1, 0), [4], dipole_tol_max=0.05)
    term = result[(1, 1, 0)][0]
    assert needed <= term["dipole_tol"] <= 0.05
    slab = term["atoms"][0]
    _assert_valid_slab(slab, Q_IRO2, _reduced(b), polarity=term["dipole_tol"])
    # The recorded tolerance lets cutslab cut the same slab.
    subs = cutslab(slab, Q_IRO2, dipole_tol=term["dipole_tol"], reconstruction=term["reconstruction"])
    assert len(subs[-1]) == len(slab)


def test_dipole_tol_max_leaves_non_polar_facets_alone():
    """The fallback runs only after a failure: facets that succeed, Tasker III
    reconstructions included, come out exactly as without it."""
    from taskerslabgen import generate_slabs_for_miller

    cases = [
        (CEO2, Q_CEO2, (0, 0, 1), {"bond_distances": BOND_DISTS_CEO2}),
        (IRO2, Q_IRO2, (1, 1, 0), {}),
        (IRO2, Q_IRO2, (1, 0, 0), {}),
    ]
    for b, q, hkl, kw in cases:
        ref = generate_slabs_for_miller(b, q, hkl, [2], candidates="all", **kw)[hkl]
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            new = generate_slabs_for_miller(
                b, q, hkl, [2], candidates="all", dipole_tol_max=0.05, **kw
            )[hkl]
        assert ref.keys() == new.keys()
        for tid in ref:
            assert new[tid]["dipole_tol"] == ref[tid]["dipole_tol"] == 1e-3
            assert new[tid]["plane_type"] == ref[tid]["plane_type"]
            assert repr(new[tid]["reconstruction"]) == repr(ref[tid]["reconstruction"])
            np.testing.assert_array_equal(new[tid]["atoms"][0].positions, ref[tid]["atoms"][0].positions)


def test_dipole_tol_max_below_dipole_tol_is_rejected():
    from taskerslabgen import generate_slabs_for_miller

    with pytest.raises(ValueError, match="dipole_tol_max"):
        generate_slabs_for_miller(IRO2, Q_IRO2, (1, 1, 0), [2], dipole_tol=0.05, dipole_tol_max=0.01)


# ------------------------------------------------------------------
# 0.5: phases -- selection="relative" / "absolute" / "shape"
# ------------------------------------------------------------------
def _ti_depth(slab):
    """Depth (A) of the topmost Ti below the topmost atom."""
    z = slab.positions[:, 2]
    return round(float(z.max() - z[slab.numbers == 22].max()), 2)


def test_termination_keeps_relative_phase_anatase101():
    """Anatase (101) has four O2 planes in one repeat unit.  With a loose
    dipole_tol, matching labels without phase mixed the second termination
    (Ti 0.15 A under the top O instead of 0.73 A) into the thickness series."""
    from taskerslabgen import cutslab, generate_slabs_for_miller

    q = {"Ti": 4.0, "O": -2.0}
    term = generate_slabs_for_miller(ANATASE, q, (1, 0, 1), [6], dipole_tol=0.05)[(1, 0, 1)][0]
    thick = term["atoms"][0]
    subs = cutslab(thick, q, dipole_tol=0.05)
    assert [len(s) for s in subs] == [12 * m for m in range(1, 7)]
    assert {_ti_depth(s) for s in subs} == {_ti_depth(thick)}
    assert {s.info["cut_top_plane"] for s in subs} == {term["top_plane_type"]}
    every_phase = cutslab(thick, q, dipole_tol=0.05, selection="shape")
    assert len(every_phase) > len(subs)
    assert {_ti_depth(s) for s in every_phase} > {_ti_depth(thick)}


def test_absolute_phase_and_phase_overlap_rutile110():
    """Rutile (110): one repeat unit up is half a cell sideways.  The top
    bridging-O row lies over the bottom one for odd layer counts only."""
    from taskerslabgen import cutslab, generate_slabs_for_miller

    term = generate_slabs_for_miller(IRO2, Q_IRO2, (1, 1, 0), [4])[(1, 1, 0)][0]
    thick = term["atoms"][0]
    relative = cutslab(thick, Q_IRO2)
    assert [len(s) for s in relative] == [6, 12, 18, 24]
    assert [round(s.info["cut_phase_overlap"]) for s in relative] == [1, 0, 1, 0]
    # absolute: same registry as the 4-layer input (top shifted from bottom)
    absolute = cutslab(thick, Q_IRO2, selection="absolute")
    assert [len(s) for s in absolute] == [12, 24]


def test_shape_selection_adds_other_phases_rutile100():
    """Rutile (100): half a repeat unit ends on another O phase (O'')."""
    from taskerslabgen import cutslab, generate_slabs_for_miller

    term = generate_slabs_for_miller(IRO2, Q_IRO2, (1, 0, 0), [4])[(1, 0, 0)][0]
    thick = term["atoms"][0]
    relative = cutslab(thick, Q_IRO2)
    assert {s.info["cut_top_plane"] for s in relative} == {term["top_plane_type"]}
    shape = cutslab(thick, Q_IRO2, selection="shape")
    assert len(shape) == 2 * len(relative)
    assert {s.info["cut_top_plane"] for s in shape} == {"O", "O''"}


def test_lattice_copies_in_a_centred_cell_share_labels():
    """Body-centred anatase: the (001) cell holds every plane twice, a lattice
    translation apart, so the labels repeat after half the cell."""
    from taskerslabgen.advanced import build_surface, identify_planes
    from taskerslabgen.core import _repeat_names, _surface_a3_xy

    surf = build_surface(ANATASE, (0, 0, 1), layers=1)
    L = surf.cell[2, 2]
    atoms_z = np.column_stack([surf.numbers, surf.positions[:, 2], np.zeros(len(surf))])
    planes = sorted(identify_planes(atoms_z, L), key=lambda p: p["z_center"])
    names, _, per = _repeat_names(planes, surf, L, _surface_a3_xy(ANATASE, (0, 0, 1), surf.cell[:2, :2]))
    assert per == len(planes) // 2
    assert names[:per] == names[per:]


def test_rotated_phases_named_independently_of_origin():
    """The two IrO2 planes of rutile (001) are one arrangement rotated by 90
    degrees; which one is primed must not depend on the bulk origin."""
    from taskerslabgen import generate_slabs_for_miller

    rng = np.random.default_rng(11)
    seen = set()
    for shift in [np.zeros(3)] + [rng.random(3) for _ in range(4)]:
        res = generate_slabs_for_miller(_shifted(IRO2, shift), Q_IRO2, (0, 0, 1), [2], candidates="all")
        seen.add(frozenset((t["plane_type"], _bottom_plane_descriptor(t["atoms"][0]))
                           for t in res[(0, 0, 1)].values()))
    assert len(seen) == 1


def test_genslab_selection_modes():
    from taskerslabgen import generate_slabs_for_miller

    res = generate_slabs_for_miller(IRO2, Q_IRO2, (1, 0, 0), [2], candidates="all")[(1, 0, 0)]
    labels = {t["plane_type"] for t in res.values()}
    base = sorted(labels)[0].rstrip("'")
    strict = generate_slabs_for_miller(IRO2, Q_IRO2, (1, 0, 0), [2], candidates="all",
                                       prefer_plane=sorted(labels)[0])[(1, 0, 0)]
    loose = generate_slabs_for_miller(IRO2, Q_IRO2, (1, 0, 0), [2], candidates="all",
                                      prefer_plane=base, selection="shape")[(1, 0, 0)]
    assert {t["plane_type"] for t in strict.values()} == {sorted(labels)[0]}
    assert {t["plane_type"] for t in loose.values()} == labels
    with pytest.raises(ValueError, match="cutslab"):
        generate_slabs_for_miller(IRO2, Q_IRO2, (1, 0, 0), [2], selection="absolute")


def test_cut_at_hint_names_both_surface_planes(ceo2_111_slab):
    from taskerslabgen import cutslab

    with pytest.raises(ValueError, match=r"cut_at=\['O4', \"O4'\"\]"):
        cutslab(ceo2_111_slab, Q_CEO2, cut_at="O4")


def test_slabs_drop_bulk_cif_metadata():
    """CIF occupancies are keyed by tags; on a slab the ASE GUI drew every
    atom with the species of tag 0."""
    from taskerslabgen import cutslab, generate_slabs_for_miller

    assert "occupancy" in IRO2.info
    slab = generate_slabs_for_miller(IRO2, Q_IRO2, (1, 1, 0), [2])[(1, 1, 0)][0]["atoms"][0]
    sub = cutslab(slab, Q_IRO2)[0]
    for atoms in (slab, sub):
        assert not {"occupancy", "spacegroup", "unit_cell"} & set(atoms.info)


def test_cuts_top_bottom_and_old_names(ceo2_111_slab):
    """cuts= names follow the vertical plots: "top" keeps the bottom plane
    and cuts from the top; "right"/"left" are the old names."""
    from taskerslabgen import cutslab

    top = cutslab(ceo2_111_slab, Q_CEO2, cuts="top")
    bottom = cutslab(ceo2_111_slab, Q_CEO2, cuts="bottom")
    assert len({s.info["cut_bottom_idx"] for s in top}) == 1
    assert len({s.info["cut_top_idx"] for s in bottom}) == 1
    with pytest.warns(DeprecationWarning, match="cuts='top'"):
        old = cutslab(ceo2_111_slab, Q_CEO2, cuts="right")
    assert [len(s) for s in old] == [len(s) for s in top]


def test_thin_genslab_slab_keeps_genslab_labels():
    """A 2-layer slab is too thin to show its repeat twice; cutslab then uses
    the labels genslab stored for one repeat unit."""
    from taskerslabgen import cutslab, generate_slabs_for_miller

    q = {"Ti": 4.0, "O": -2.0}
    term = generate_slabs_for_miller(ANATASE, q, (1, 0, 1), [2], dipole_tol=0.05)[(1, 0, 1)][0]
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*repeat unit.*")
        subs = cutslab(term["atoms"][0], q, dipole_tol=0.05)
    assert {(s.info["cut_bottom_plane"], s.info["cut_top_plane"]) for s in subs} == {
        (term["plane_type"], term["top_plane_type"])}
    assert [len(s) for s in subs] == [12, 24]


def test_genslab_plot_shows_the_cut(tmp_path):
    """genslab plots the slab inside one repeat unit of bulk on each side."""
    from taskerslabgen import generate_slabs_for_miller

    generate_slabs_for_miller(IRO2, Q_IRO2, (1, 1, 0), [2], plot=True, plot_out_dir=str(tmp_path))
    generate_slabs_for_miller(CEO2, Q_CEO2, (0, 0, 1), [2], prefer_plane="O",
                              bond_distances=BOND_DISTS_CEO2, plot=True, plot_out_dir=str(tmp_path))
    assert len(list(tmp_path.glob("*.png"))) == 2


@pytest.mark.parametrize(
    "atoms, charges, hkl",
    [(IRO2, Q_IRO2, (1, 0, 0)), (ANATASE, {"Ti": 4.0, "O": -2.0}, (0, 0, 1))],
    ids=["IrO2100", "anatase001"],
)
def test_bulk_mode_keeps_phases_of_glide_related_terminations(atoms, charges, hkl):
    """A glide maps these stackings onto themselves half a repeat up, so two
    height registries fit the bulk equally well plane by plane; only one
    aligns every plane with a single in-plane translation.  Bulk mode named
    termination O'''/O'' of rutile (100) O'/O."""
    from taskerslabgen import cutslab, generate_slabs_for_miller

    for term in generate_slabs_for_miller(atoms, charges, hkl, [3], candidates="all")[hkl].values():
        subs = cutslab(term["atoms"][0], charges, bulk_atoms=atoms)
        assert {(s.info["cut_bottom_plane"], s.info["cut_top_plane"]) for s in subs} == {
            (term["plane_type"], term["top_plane_type"])}


def test_bulk_mode_robust_to_noise_where_slab_mode_warns():
    """Anatase (101) Ti2 and O2' are 0.15 A apart: 0.03 A noise on every atom
    blurs them for height clustering (cutslab warns), not for bulk mode."""
    from taskerslabgen import cutslab, generate_slabs_for_miller

    q = {"Ti": 4.0, "O": -2.0}
    term = generate_slabs_for_miller(ANATASE, q, (1, 0, 1), [4])[(1, 0, 1)][0]
    noisy = term["atoms"][0].copy()
    noisy.positions += np.random.default_rng(1).normal(0.0, 0.03, noisy.positions.shape)
    subs = cutslab(noisy, q, dipole_tol=0.05, bulk_atoms=ANATASE)
    assert [len(s) for s in subs] == [12, 24, 36, 48]
    assert {(s.info["cut_bottom_plane"], s.info["cut_top_plane"]) for s in subs} == {
        (term["plane_type"], term["top_plane_type"])}
    with pytest.warns(UserWarning, match="repeat unit"):
        cutslab(noisy, q, dipole_tol=0.05)


@pytest.mark.parametrize(
    "atoms, charges, hkl",
    [(ANATASE, {"Ti": 4.0, "O": -2.0}, (0, 0, 1)), (ALBITE, Q_ALBITE, (0, 0, 1))],
    ids=["anatase001", "albite001"],
)
def test_slab_labels_without_genslab_hint_group_planes_alike(atoms, charges, hkl):
    """cutslab names a slab's planes from its own interior when genslab's
    labels are not stored; the names may differ, the grouping may not.  The
    reference repeat unit used planes from before its window (moved in-plane
    by the repeat translation), merging two O phases of anatase (001)."""
    from taskerslabgen import generate_slabs_for_miller
    from taskerslabgen.advanced import identify_planes
    from taskerslabgen.core import _slab_plane_names

    for term in generate_slabs_for_miller(atoms, charges, hkl, [4], candidates="all")[hkl].values():
        slab = term["atoms"][0]
        atoms_z = np.column_stack([slab.numbers, slab.positions[:, 2], np.zeros(len(slab))])
        planes = sorted(identify_planes(atoms_z, slab.cell[2, 2]), key=lambda p: p["z_center"])
        hinted, _ = _slab_plane_names(slab, planes, 2, slab.info["stacking_labels"])
        own, repeat = _slab_plane_names(slab, planes, 2, None)
        assert repeat is not None
        pairs = set(zip(hinted, own))
        assert len(pairs) == len(set(hinted)) == len(set(own)), sorted(pairs)


def test_old_ase_with_numpy2_fails_with_a_clear_message(monkeypatch):
    """ASE 3.22 calls numpy.product, gone in NumPy 2: say so at import."""
    import ase

    import taskerslabgen

    monkeypatch.setattr(np, "__version__", "2.1.0")
    monkeypatch.setattr(ase, "__version__", "3.22.1")
    with pytest.raises(ImportError, match="ASE >= 3.23"):
        taskerslabgen._check_ase_numpy()
    monkeypatch.setattr(ase, "__version__", "3.29.0")
    taskerslabgen._check_ase_numpy()
