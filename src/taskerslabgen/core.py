"""
Shared machinery: surface cells, plane clustering and labelling (arrangement and
phase, from smooth plane densities), cut enumeration, polarity, validation and
matching slab planes to the bulk.
"""
import re
from collections import Counter
from functools import cmp_to_key
from itertools import product
from math import gcd

import numpy as np
from ase.build import surface
from ase.data import atomic_numbers, chemical_symbols
from ase.formula import Formula
from scipy.optimize import linear_sum_assignment

# Single-linkage z-gap (angstrom) below which atoms share a plane; same
# default as pymatgen's ``ftol``.
DEFAULT_PLANE_TOL = 0.1

# Polarity is measured as the dipole per surface area, with the charges
# divided by their mean absolute value: any proportional set of charges
# (formal, relative, computed) gives the same numbers, and a slab's verdict
# does not depend on its thickness.  Units: 1/angstrom.
DEFAULT_DIPOLE_TOL = 1e-3


def _charge_scale(charges):
    """Mean absolute charge per atom: the unit in which charges are compared."""
    q = np.abs(np.asarray(charges, dtype=float))
    scale = float(q.mean()) if q.size else 0.0
    return scale if scale > 0 else 1.0


def _surface_area(cell, axis=2):
    """Area (angstrom^2) of the cell face perpendicular to *axis*."""
    a, b = [np.asarray(cell[i], dtype=float) for i in range(3) if i != axis]
    return float(np.linalg.norm(np.cross(a, b)))


def dipole_per_area(charges, positions, cell, axis=2, charge_scale=None):
    """
    Polarity of a slab: |dipole along *axis*| per surface area, with the
    charges divided by *charge_scale* (default: their mean absolute value).
    In 1/angstrom; this is what ``dipole_tol`` is compared with.

    Parameters
    ----------
    charges : array-like
        Charge of every atom (any scale: formal, relative or computed).
    positions : array-like, shape (n, 3)
        Cartesian positions (angstrom), e.g. ``slab.positions``.
    cell : array-like, shape (3, 3)
        Cell vectors; the surface area is that of the two vectors other
        than *axis*.
    axis : int
        Surface normal (cell vector index and Cartesian axis).
    charge_scale : float or None
        Divide the charges by this instead of their mean absolute value.

    Example: ``dipole_per_area([charges[s] for s in slab.get_chemical_symbols()],
    slab.positions, slab.cell)``.
    """
    q = np.asarray(charges, dtype=float)
    scale = _charge_scale(q) if charge_scale is None else charge_scale
    z = np.asarray(positions, dtype=float)[:, axis]
    mu = float(np.sum(q * (z - z.mean())))
    return abs(mu) / (_surface_area(cell, axis) * scale)


# Per-atom array used internally to track which input atom each slab atom
# came from (survives ``ase.build.surface`` and slicing).  Removed before
# slabs are returned.
_INDEX_KEY = "_tsg_index"


class SlabValidationError(ValueError):
    """A generated slab is not stoichiometric, neutral, non-polar and contiguous."""


class PolarSurfaceError(ValueError):
    """
    No termination or reconstruction of a facet is non-polar within
    ``dipole_tol``.

    ``min_dipole`` is the smallest polarity (dipole per surface area,
    charges normalised, 1/Å; see :func:`dipole_per_area`) that any candidate
    reaches over the requested thicknesses: the ``dipole_tol`` a slab would
    need.
    """

    def __init__(self, message, min_dipole):
        super().__init__(message)
        self.min_dipole = float(min_dipole)


def _check_miller(miller):
    hkl = tuple(int(round(x)) for x in miller)
    if len(hkl) != 3 or any(abs(x - y) > 1e-9 for x, y in zip(miller, hkl)):
        raise ValueError(f"Miller index must be three integers, got {miller!r}.")
    if not any(hkl):
        raise ValueError("Miller index (0, 0, 0) does not define a plane.")
    g = gcd(gcd(abs(hkl[0]), abs(hkl[1])), abs(hkl[2]))
    if g != 1:
        reduced = tuple(x // g for x in hkl)
        raise ValueError(
            f"Miller index {hkl} is not reduced; use {reduced} (same surface orientation)."
        )


def _tag_atom_indices(atoms):
    """Return a copy of *atoms* that records each atom's index in ``_INDEX_KEY``."""
    tagged = atoms.copy()
    tagged.arrays[_INDEX_KEY] = np.arange(len(tagged))
    return tagged


def build_surface(bulk_atoms, miller, layers=1, vacuum=None, verbose=None):
    """
    Build an ASE surface slab from a bulk structure.

    Parameters
    ----------
    bulk_atoms : Atoms
        Bulk unit cell.
    miller : tuple of int
        Miller index ``(h, k, l)``.
    layers : int
        Number of bulk repeat units along the surface normal.
    vacuum : float or None
        Vacuum to add (angstrom, per side).  ``None`` or ``0`` (default)
        returns the bare oriented bulk, whose third cell vector is normal to
        the surface with length ``layers * L``.
    verbose : bool or None
        Print debug information.

    Returns
    -------
    Atoms
        Surface slab with full PBC enabled.  Atoms of ``ase.build.surface``
        are moved by bulk lattice vectors (and all together along the
        normal) so that the cell boundary lies in the widest atom-free gap
        along the normal: no atomic plane straddles it, and copy ``m`` of
        an atom sits ``m * L`` above copy 0.  The third cell vector is not a
        bulk lattice vector in general; see :func:`surface_bulk_cell`.
    """
    _check_miller(miller)
    # Not ``periodic=True``: ASE would then wrap atoms along the normal by its
    # orthogonalised third vector, which is not a lattice vector.  ASE keeps
    # each atom's fractional coordinate along its oblique a3 in [0, 1), so a
    # plane can straddle z = 0 with its two halves taken from different
    # layers, which shears its in-plane geometry.  Instead, cut where the
    # gap between atoms is widest and move what lies below by whole bulk
    # lattice vectors.
    a3 = surface_bulk_cell(bulk_atoms, miller)[2]
    L = float(a3[2])
    # From the one-layer build for every *layers*: equally wide gaps (e.g.
    # rutile (110)) must not be told apart by rounding noise.
    z0 = np.sort(surface(bulk_atoms, miller, layers=1).positions[:, 2] % L)
    gaps = np.diff(np.append(z0, z0[0] + L))
    k = int(np.flatnonzero(gaps >= gaps.max() - 1e-6)[0])
    z_cut = (z0[k] + 0.5 * gaps[k]) % L
    slab = surface(bulk_atoms, miller, layers=layers)
    pos = slab.positions.copy()
    pos[pos[:, 2] < z_cut] += layers * a3
    pos[:, 2] -= z_cut
    slab.positions = pos
    slab.cell[2] = [0.0, 0.0, layers * L]
    frac = slab.get_scaled_positions(wrap=False)
    frac[:, :2] %= 1.0
    slab.set_scaled_positions(frac)
    if vacuum is not None and vacuum > 0:
        slab.center(vacuum=vacuum, axis=2)
    slab.set_pbc((True, True, True))
    if verbose:
        print("BULK")
        print(bulk_atoms, bulk_atoms.positions, "\n")
        print("REORIENTED BULK")
        print(slab, slab.positions, "\n")
    return slab


def compute_projection(bulk, surf_bulk, charges, miller, verbose=None):
    """
    Compute the z-projection matrix ``[Z, z_coord, charge]`` for each atom.

    Parameters
    ----------
    bulk : Atoms
        Original bulk cell (used to compute the lattice-plane spacing *L*).
    surf_bulk : Atoms
        Reoriented 1-layer surface slab.
    charges : dict, list or None
        Formal charges.  A dict maps element symbols or atomic numbers to
        charge values; a list gives per-atom charges; ``None`` uses the
        charges stored on *surf_bulk* (see :func:`generate_slabs_for_miller`).
    miller : tuple of int
        Miller index ``(h, k, l)``.
    verbose : bool or None
        Print debug information.

    Returns
    -------
    atoms_z_matrix : ndarray, shape (N, 3)
        Each row is ``[atomic_number, z_position, charge]``.
    L : float
        Lattice-plane spacing (angstrom) for this Miller index.
    """
    _check_miller(miller)
    if charges is None or isinstance(charges, dict):
        charges = _charges_to_list(surf_bulk, charges)
    if len(charges) != len(surf_bulk):
        raise ValueError(
            f"Charges length ({len(charges)}) does not match atoms ({len(surf_bulk)})."
        )
    cell = bulk.cell
    recip = cell.reciprocal()
    hkl = np.array(miller, dtype=float)
    G = hkl @ recip
    L = 1.0 / np.linalg.norm(G)
    if L <= 0.0:
        raise ValueError("Invalid cell height along z.")
    z_coords = surf_bulk.positions[:, 2]
    atoms_z_matrix = np.array(
        [[num, z, q] for num, z, q in zip(surf_bulk.numbers, z_coords, charges)]
    )
    if verbose:
        print("Atom matrix [Z, z, q]:")
        print(atoms_z_matrix, "\n")
    return atoms_z_matrix, L


def surface_bulk_cell(bulk_atoms, miller):
    """
    Periodic cell of the bulk crystal in the frame of :func:`build_surface`.

    ``ase.build.surface`` replaces the third lattice vector by one normal to
    the surface.  That vector is generally *not* a bulk lattice vector, so
    periodic images across layers computed with it are wrong.  The true
    third vector is the displacement between the two copies of the same atom
    in a two-layer build.

    Returns
    -------
    ndarray, shape (3, 3)
        Rows ``a1, a2`` (in-plane, as in :func:`build_surface`) and the bulk
        lattice vector ``a3`` that stacks one layer onto the next.
    """
    _check_miller(miller)
    two = surface(bulk_atoms, miller, layers=2)
    n = len(bulk_atoms)
    a3 = two.positions[n] - two.positions[0]
    return np.array([two.cell[0], two.cell[1], a3])


def _oriented_bulk(bulk_atoms, miller, surface_supercell=(1, 1)):
    """
    The bulk crystal in the frame of :func:`build_surface`, as a cell whose
    ``(0, 0, 1)`` surface is the *miller* surface of *bulk_atoms*, repeated
    ``surface_supercell = (n1, n2)`` times in-plane.

    Rows of the cell are the in-plane vectors of :func:`build_surface` and
    the true bulk vector ``a3`` (:func:`surface_bulk_cell`); per-atom arrays
    are kept.
    """
    try:
        n1, n2 = (int(x) for x in surface_supercell)
        ok = (n1, n2) == tuple(surface_supercell) and n1 >= 1 and n2 >= 1
    except (TypeError, ValueError):
        ok = False
    if not ok:
        raise ValueError(
            f"surface_supercell must be two positive integers (n1, n2), got {surface_supercell!r}."
        )
    oriented = build_surface(bulk_atoms, miller, layers=1)
    oriented.set_cell(surface_bulk_cell(bulk_atoms, miller), scale_atoms=False)
    return oriented.repeat((n1, n2, 1))


def _charges_from_atoms(atoms):
    """Per-atom charges stored on *atoms*: calculator results, then initial charges."""
    calc = getattr(atoms, "calc", None)
    results = getattr(calc, "results", None) or {}
    if results.get("charges") is not None:
        return [float(q) for q in results["charges"]]
    if atoms.has("initial_charges") and np.any(atoms.get_initial_charges()):
        return [float(q) for q in atoms.get_initial_charges()]
    raise ValueError(
        "charges=None needs per-atom charges on the Atoms object (calculator results "
        "'charges' or atoms.set_initial_charges(...)).  For FHI-aims Hirshfeld charges "
        "use atoms.set_initial_charges(parse_hirshfeld_fhi_aims(path)); otherwise pass "
        "charges as a dict, e.g. {'Ce': 4.0, 'O': -2.0}."
    )


def _charges_to_list(atoms, charges):
    """
    Per-atom charges of *atoms*.

    *charges* is a dict ``{symbol or Z: charge}``, a per-atom list, or
    ``None`` to use the charges stored on *atoms* (calculator results
    ``"charges"``, e.g. from an extxyz file, else ``initial_charges``).
    """
    if charges is None:
        return _charges_from_atoms(atoms)
    if isinstance(charges, dict):
        charge_map = {}
        for key, val in charges.items():
            if isinstance(key, str):
                if key not in atomic_numbers:
                    raise ValueError(f"Unknown element symbol: {key}")
                charge_map[atomic_numbers[key]] = float(val)
            elif isinstance(key, int):
                charge_map[key] = float(val)
            else:
                raise ValueError(f"Unsupported charge key type: {type(key)}")
        charges_list = []
        for Z in atoms.numbers:
            if Z not in charge_map:
                raise ValueError(f"Missing charge for atomic number: {Z}")
            charges_list.append(charge_map[Z])
        return charges_list
    return list(charges)


def _make_plane(indices, z_values, atoms_z, charge_tol, period=None):
    """Plane dict of the atoms *indices* at heights *z_values* (unwrapped)."""
    z_values = np.asarray(z_values, dtype=float)
    z_mean = float(np.mean(z_values))
    q = np.asarray(atoms_z[indices, 2], dtype=float)
    q_total = float(np.sum(q))
    if abs(q_total) < charge_tol:
        q_total = 0.0
    counts = {}
    for Z in atoms_z[indices, 0].astype(int):
        counts[int(Z)] = counts.get(int(Z), 0) + 1
    return {
        "z_center": z_mean % period if period else z_mean,
        "z_lo": float(z_values.min()) - z_mean,
        "z_hi": float(z_values.max()) - z_mean,
        "q_total": q_total,
        "dipole": float(np.sum(q * (z_values - z_mean))),
        "indices": [int(i) for i in indices],
        "counts": counts,
    }


def identify_planes(atoms_z, L, plane_tol=None, charge_tol=1e-3):
    """
    Cluster atoms into atomic planes along the stacking direction.

    Single-linkage clustering of the z-coordinates (mod *L*), as in
    pymatgen's ``SlabGenerator``: two atoms share a plane when they are
    joined by a chain of neighbours whose z-gaps are all at most
    *plane_tol*.  The result does not depend on atom order, and planes that
    straddle the periodic boundary are kept whole.

    Parameters
    ----------
    atoms_z : ndarray, shape (N, 3)
        ``[atomic_number, z_position, charge]`` matrix.
    L : float
        Lattice-plane spacing (angstrom).
    plane_tol : float or None
        Largest z-gap (angstrom) between neighbouring atoms of one plane.
        ``None`` (default) uses ``DEFAULT_PLANE_TOL`` (0.1 Å).
    charge_tol : float
        Plane charges with ``abs(q) < charge_tol`` are set to exactly 0.

    Returns
    -------
    list of dict
        Planes sorted by ``z_center`` (in ``[0, L)``).  Each dict contains
        ``z_center``, ``z_lo`` / ``z_hi`` (lowest / highest atom relative to
        ``z_center``), ``q_total``, ``dipole`` (``sum q (z - z_center)`` of
        its atoms), ``indices``, and ``counts`` (element composition
        ``{Z: count}``).
    """
    if len(atoms_z) == 0:
        return []

    tol = DEFAULT_PLANE_TOL if plane_tol is None else float(plane_tol)
    L = float(L)
    z_mod = np.asarray(atoms_z[:, 1], dtype=float) % L
    order = np.argsort(z_mod, kind="stable")
    z_sorted = z_mod[order]
    n = len(order)

    # gaps[k] separates sorted atoms k and k+1; the last gap wraps through L.
    gaps = np.diff(np.concatenate([z_sorted, [z_sorted[0] + L]]))
    breaks = np.flatnonzero(gaps > tol)
    if len(breaks) == 0:
        # One plane: unwrap it across its widest gap.
        breaks = np.array([int(np.argmax(gaps))])
        break_set = set()
    else:
        break_set = {int(b) for b in breaks}

    # Walk the atoms cyclically starting just after a break so no plane is
    # split by the periodic boundary; unwrap z on the way.
    start = (int(breaks[-1]) + 1) % n
    planes = []
    group, group_z = [], []
    for m in range(n):
        k = (start + m) % n
        group.append(order[k])
        group_z.append(z_sorted[k] + (L if start + m >= n else 0.0))
        if k in break_set or m == n - 1:
            planes.append(_make_plane(group, group_z, atoms_z, charge_tol, period=L))
            group, group_z = [], []

    planes.sort(key=lambda p: p["z_center"])
    return planes


def compute_reduced_counts(atoms_z):
    """
    Compute the reduced (primitive) stoichiometry of the unit cell.

    Parameters
    ----------
    atoms_z : ndarray, shape (N, 3)
        ``[atomic_number, z_position, charge]`` matrix.

    Returns
    -------
    dict
        ``{atomic_number: reduced_count}`` with the GCD factored out.
    """
    types = np.unique(atoms_z[:, 0].astype(int))
    counts = {Z: int(np.sum(atoms_z[:, 0] == Z)) for Z in types}
    gcd = 0
    for c in counts.values():
        gcd = np.gcd(gcd, c)
    gcd = max(int(gcd), 1)
    reduced = {Z: counts[Z] // gcd for Z in types}
    return reduced


def is_stoichiometric_sequence(sequence_counts, reduced_counts):
    """
    Check whether a plane sequence has an integer multiple of the bulk
    stoichiometry.

    Parameters
    ----------
    sequence_counts : dict
        ``{atomic_number: count}`` for the sequence of planes.
    reduced_counts : dict
        Reduced bulk stoichiometry from :func:`compute_reduced_counts`.

    Returns
    -------
    is_stoich : bool
        True if the sequence is a whole-number multiple of the bulk formula.
    k : int or None
        The multiplier, or None if not stoichiometric.
    """
    ks = []
    for Z, reduced in reduced_counts.items():
        if reduced == 0:
            continue
        count = sequence_counts.get(Z, 0)
        if count % reduced != 0:
            return False, None
        ks.append(count // reduced)
    if not ks:
        return False, None
    if len(set(ks)) != 1:
        return False, None
    if ks[0] < 1:
        return False, None
    return True, ks[0]


def enumerate_cut_pairs(planes, L, reduced_counts, charge_tol=1e-3, *, area, charge_scale):
    """
    Enumerate all contiguous plane sequences and compute their charge,
    stoichiometry, and dipole moment.

    Parameters
    ----------
    planes : list of dict
        Plane dicts from :func:`identify_planes`.
    L : float
        Lattice-plane spacing (angstrom).
    reduced_counts : dict
        Reduced bulk stoichiometry.
    charge_tol : float
        Largest |net charge| per formula unit, in units of *charge_scale*,
        treated as neutral.
    area : float
        Surface area of the cell (angstrom^2), e.g. ``_surface_area(surf.cell)``.
    charge_scale : float
        Mean absolute charge per atom of the bulk, e.g.
        ``_charge_scale(atoms_z[:, 2])``.  Required, like *area*, so that
        polarities are always normalised.

    Returns
    -------
    list of dict
        Each entry describes a cut sequence with keys ``bottom_cut``,
        ``top_cut``, ``plane_indices``, ``total_charge``, ``net_dipole``
        (e·Å), ``is_neutral``, ``is_stoich``, ``stoich_k``,
        ``dipole_per_area`` (|dipole| / (area * charge_scale), 1/Å, ``None``
        if not stoichiometric), ``is_full_period`` (the sequence spans one
        whole bulk repeat unit), etc.
    """
    if len(planes) == 0:
        return []

    planes_sorted = sorted(planes, key=lambda p: p["z_center"] % L)
    z_sorted = np.array([p["z_center"] % L for p in planes_sorted], dtype=float)
    q_sorted = np.array([p["q_total"] for p in planes_sorted], dtype=float)
    # Dipole of each plane's own atoms about its centre: thick or rumpled
    # planes are not point charges.
    p_sorted = np.array([p.get("dipole", 0.0) for p in planes_sorted], dtype=float)
    counts_sorted = [p["counts"] for p in planes_sorted]
    n = len(planes_sorted)
    atoms_per_fu = float(sum(reduced_counts.values()))

    sequences = []
    for bottom_cut in range(n):
        for top_cut in range(n):
            bottom_start = (bottom_cut + 1) % n
            top_end = top_cut

            seq_indices_btt = []
            idx = bottom_start
            while True:
                seq_indices_btt.append(idx)
                if idx == top_end:
                    break
                idx = (idx + 1) % n

            z_seq_btt = []
            z_current = float(z_sorted[seq_indices_btt[0]])
            z_seq_btt.append(z_current)
            for i in seq_indices_btt[1:]:
                z_next = float(z_sorted[i])
                if z_next < z_current:
                    z_next += L
                z_seq_btt.append(z_next)
                z_current = z_next
            z_seq_btt = np.array(z_seq_btt, dtype=float)
            q_seq_btt = np.array([q_sorted[i] for i in seq_indices_btt], dtype=float)

            seq_counts = {}
            for i in seq_indices_btt:
                for Z, c in counts_sorted[i].items():
                    seq_counts[Z] = seq_counts.get(Z, 0) + c
            is_stoich, stoich_k = is_stoichiometric_sequence(seq_counts, reduced_counts)
            total_q = float(np.sum(q_seq_btt))
            z_center_btt = 0.5 * (float(z_seq_btt[0]) + float(z_seq_btt[-1]))
            mu_btt = float(np.sum(q_seq_btt * (z_seq_btt - z_center_btt))
                           + np.sum(p_sorted[seq_indices_btt]))
            sequences.append(
                {
                    "bottom_cut": bottom_cut,
                    "top_cut": top_cut,
                    "plane_indices": seq_indices_btt,
                    "total_charge": total_q,
                    "net_dipole": mu_btt,
                    "z_center": z_center_btt,
                    "direction": "bottom-to-top",
                    "plane_z": [float(z % L) for z in z_seq_btt],
                    "plane_Q": [float(q) for q in q_seq_btt],
                    "is_neutral": abs(total_q) / charge_scale <= charge_tol * max(
                        1.0, sum(seq_counts.values()) / atoms_per_fu),
                    "is_stoich": is_stoich,
                    "stoich_k": stoich_k,
                    "dipole_per_area": (abs(mu_btt) / (area * charge_scale)
                                        if is_stoich else None),
                    "is_full_period": len(seq_indices_btt) == n,
                }
            )

    sequences.sort(key=lambda s: abs(s["net_dipole"]), reverse=True)
    return sequences


def select_best_sequence(sequences, dipole_tol=DEFAULT_DIPOLE_TOL, n_units=1):
    """
    Select the best stoichiometric, charge-neutral sequence (lowest dipole).

    Only full-period sequences (one bulk repeat unit along the normal) are
    considered: they are what slabs are stacked from, so they decide the
    Tasker type.  A zero-dipole partial sequence whose remainder is polar
    cannot be stacked into a thick non-polar slab.  Among zero-dipole
    sequences the one with the lowest ``bottom_cut`` is returned, so the
    choice does not depend on floating-point noise.

    Parameters
    ----------
    sequences : list of dict
        Output of :func:`enumerate_cut_pairs`.
    dipole_tol : float
        Largest polarity (dipole per surface area, charges normalised, 1/Å)
        still considered zero (Tasker I/II).
    n_units : int
        Repeat units of the thickest slab to be built: stacked units add
        their dipoles, so a slab of *n_units* has *n_units* times the
        polarity of one.

    Returns
    -------
    dict or None
        Copy of the best sequence with an added ``is_tasker_ii`` flag, or
        None.
    """
    valid = [
        s for s in sequences
        if s["is_neutral"] and s["is_stoich"] and s.get("is_full_period", True)
    ]
    if not valid:
        return None

    def key(s):
        d = s["dipole_per_area"] * n_units
        return (0.0 if d <= dipole_tol else d, s["bottom_cut"])

    best = dict(min(valid, key=key))
    best["is_tasker_ii"] = best["dipole_per_area"] * n_units <= dipole_tol
    return best


def compute_cut_positions(planes, L, bottom_cut_index, top_cut_index):
    """
    Compute z-coordinates for the bottom and top cuts.

    Each cut lies in the middle of the empty gap between the highest atom of
    one plane and the lowest atom of the next (``z_lo`` / ``z_hi`` from
    :func:`identify_planes`; plane centres if absent), so no atom of a thick
    plane is cut off.

    Parameters
    ----------
    planes : list of dict
        Plane dicts from :func:`identify_planes`.
    L : float
        Lattice-plane spacing (angstrom).
    bottom_cut_index : int
        Index of the plane *below* the bottom cut.
    top_cut_index : int
        Index of the plane *above* the top cut.

    Returns
    -------
    zbot : float
        z-coordinate of the bottom cut.
    ztop : float
        z-coordinate of the top cut.
    """
    planes_sorted = sorted(planes, key=lambda p: p["z_center"] % L)
    n = len(planes_sorted)

    def midpoint(i):
        p0, p1 = planes_sorted[i], planes_sorted[(i + 1) % n]
        z0, z1 = p0["z_center"] % L, p1["z_center"] % L
        if z1 <= z0:  # wraps (or a single plane: next copy is one L above)
            z1 += L
        return 0.5 * ((z0 + p0.get("z_hi", 0.0)) + (z1 + p1.get("z_lo", 0.0)))

    return midpoint(bottom_cut_index), midpoint(top_cut_index)


def apply_vacuum_to_slab(atoms, vacuum=15.0, axis=2):
    """
    Add vacuum above and below a slab by shifting atoms and resizing the cell.

    Atoms are shifted so the bottom of the slab sits at 0 along the given axis,
    and the cell is extended by `vacuum` angstrom on each side (top and bottom).
    """
    if vacuum <= 0:
        return
    positions = atoms.get_positions()
    z_positions = positions[:, axis]
    zmin = float(np.min(z_positions))
    zmax = float(np.max(z_positions))
    new_zmin = zmin - vacuum
    new_zmax = zmax + vacuum
    new_height = new_zmax - new_zmin
    positions[:, axis] -= new_zmin
    # A rigid shift of the whole slab; FixAtoms must not hold atoms back.
    atoms.set_positions(positions, apply_constraint=False)
    cell = atoms.get_cell().copy()
    vec = np.zeros(3)
    vec[axis] = new_height
    cell[axis] = vec
    atoms.set_cell(cell)
    atoms.set_pbc([True, True, True])


def validate_slab(slab, charges, reduced_counts, axis=2, charge_tol=1e-3,
                  dipole_tol=DEFAULT_DIPOLE_TOL, max_gap=None):
    """
    Check that *slab* satisfies the conditions the generators promise.

    Parameters
    ----------
    slab : Atoms
        Slab to check.
    charges : dict, list or None
        Charges by element, one charge per atom of *slab*, or ``None`` for
        the charges stored on *slab*.
    reduced_counts : dict
        Reduced bulk stoichiometry ``{Z: count}``.
    axis : int
        Surface-normal axis.
    charge_tol : float
        Largest net charge per formula unit, in units of the mean absolute
        charge per atom.
    dipole_tol : float
        Largest polarity: |dipole along *axis*| per surface area with the
        charges divided by their mean absolute value, in 1/Å
        (:func:`dipole_per_area`).  Independent of the thickness and of the
        scale of the charges.
    max_gap : float or None
        Largest allowed z-gap (angstrom) between neighbouring atoms; catches
        slabs glued together across vacuum.  ``None`` (default) skips the
        check.  Not used by the generators: deleting atoms from a rumpled
        Tasker III surface plane legitimately widens a gap.

    Raises
    ------
    SlabValidationError
        If the slab is not stoichiometric, not neutral, polar, or has an
        internal gap larger than *max_gap*.  The message lists every failed
        check.
    """
    problems = []
    counts = {}
    for Z in slab.numbers:
        counts[int(Z)] = counts.get(int(Z), 0) + 1
    is_stoich, k = is_stoichiometric_sequence(counts, reduced_counts)
    if not is_stoich or set(counts) - set(reduced_counts):
        problems.append("not stoichiometric")
        k = max(1, round(len(slab) / max(1, sum(reduced_counts.values()))))

    q = np.asarray(_charges_to_list(slab, charges), dtype=float)
    if len(q) != len(slab):
        raise ValueError(f"Charges length ({len(q)}) does not match atoms ({len(slab)}).")
    scale = _charge_scale(q)
    net_q = float(np.sum(q))
    if abs(net_q) > charge_tol * k * scale:
        problems.append(f"net charge {net_q:+.4f} e")

    z = slab.positions[:, axis]
    polarity = dipole_per_area(q, slab.positions, slab.cell, axis, scale)
    if polarity > dipole_tol:
        problems.append(f"dipole {float(np.sum(q * (z - z.mean()))):+.4e} e*A "
                        f"(polarity {polarity:.3g} /A)")

    if max_gap is not None and len(slab) > 1:
        gap = float(np.max(np.diff(np.sort(z))))
        if gap > max_gap + 1e-6:
            problems.append(f"internal z-gap of {gap:.2f} A (bulk max {max_gap:.2f} A)")

    if problems:
        raise SlabValidationError(
            f"Invalid slab {slab.get_chemical_formula()}: {'; '.join(problems)} "
            f"(charge_tol={charge_tol}, dipole_tol={dipole_tol} /A)."
        )


# Bulk metadata from CIF files that is wrong for a slab: the bulk space
# group, and site occupancies keyed by tags (the ASE GUI draws atoms from
# them, so a slab would be drawn with the bulk's species).
_BULK_ONLY_INFO = ("spacegroup", "unit_cell", "occupancy")


def _finalize_slab(slab, charges_list, reduced_counts, charge_tol, dipole_tol,
                   max_gap=None, axis=2):
    """Validate a slab cut from an index-tagged structure, then drop the tag
    and the bulk-only metadata."""
    q = np.asarray(charges_list, dtype=float)[slab.arrays[_INDEX_KEY]]
    del slab.arrays[_INDEX_KEY]
    for key in _BULK_ONLY_INFO:
        slab.info.pop(key, None)
    validate_slab(slab, q, reduced_counts, axis=axis, charge_tol=charge_tol,
                  dipole_tol=dipole_tol, max_gap=max_gap)


# Plane "waves": each plane is a smooth periodic density, one Gaussian of
# width _WAVE_SIGMA per atom and one channel per element, summed over all
# in-plane periodic images.  Two planes are compared by the overlap of their
# densities (normalised to 1 for identical planes).
_WAVE_SIGMA = 0.5   # angstrom; how smooth the density is, not a cutoff
_SAME_PLANE = 0.9   # normalised overlap from which two planes count as the same


def _wave_rows(geom):
    """Plane atoms ``[(Z, fx, fy[, dz]), ...]`` as an ``(n, 4)`` float array."""
    rows = np.zeros((len(geom), 4))
    for k, atom in enumerate(geom):
        rows[k, :len(atom[:4])] = atom[:4]
    return rows


def _wave_images(cell2d, sigma=_WAVE_SIGMA, eps=1e-12):
    """In-plane lattice translations (integer pairs) needed for the overlap
    sum to converge to *eps*: beyond them a Gaussian pair contributes less."""
    cell2d = np.asarray(cell2d, dtype=float)
    reach = np.sqrt(-4.0 * sigma ** 2 * np.log(eps))
    area = abs(np.linalg.det(cell2d))
    lengths = np.linalg.norm(cell2d, axis=1)
    n = [int(np.ceil(reach * lengths[1 - k] / area)) + 1 for k in range(2)]
    return np.array([(i, j) for i in range(-n[0], n[0] + 1) for j in range(-n[1], n[1] + 1)],
                    dtype=float)


def _wave_overlap(A, B, cell2d, t=(0.0, 0.0), W=None, sigma=_WAVE_SIGMA, images=None):
    """
    Overlap integral of the densities of planes *A* and *B* (rows
    ``(Z, fx, fy, dz)``), with *A* moved by the lattice point operation *W*
    (``f -> f @ W``) and then the fractional shift *t*.  Only atoms of the
    same element overlap.  Proportional to the integral of the product of the
    two densities.
    """
    cell2d = np.asarray(cell2d, dtype=float)
    images = _wave_images(cell2d, sigma) if images is None else images
    fa = A[:, 1:3] if W is None else A[:, 1:3] @ np.asarray(W, dtype=float)
    fa = fa + np.asarray(t, dtype=float)
    total = 0.0
    for Z in np.unique(A[:, 0]):
        ia, ib = A[:, 0] == Z, B[:, 0] == Z
        if not ib.any():
            continue
        d = fa[ia][:, None, :] - B[ib][None, :, 1:3]
        d -= np.round(d)
        cart = (d[:, :, None, :] + images[None, None, :, :]) @ cell2d
        dz = (A[ia][:, None, 3] - B[ib][None, :, 3])[:, :, None]
        total += float(np.exp(-(np.sum(cart ** 2, axis=-1) + dz ** 2) / (4.0 * sigma ** 2)).sum())
    return total


def _wave_similarity(A, B, cell2d, t=(0.0, 0.0), W=None, images=None):
    """Normalised overlap in [0, 1]: 1 when *B* is *A* moved by *W* and *t*."""
    images = _wave_images(cell2d) if images is None else images
    norm = np.sqrt(_wave_overlap(A, A, cell2d, images=images)
                   * _wave_overlap(B, B, cell2d, images=images))
    return _wave_overlap(A, B, cell2d, t, W, images=images) / norm if norm > 0 else 0.0


def _wave_alignment(A, B, cell2d, ops=None, images=None):
    """
    Best overlap of plane *A* onto plane *B* over the point operations *ops*
    of the in-plane lattice (identity only if None) and all shifts.

    Returns ``(similarity, W, t)``.  Shifts start from putting an atom of the
    rarest element of *A* on each atom of that element in *B* and are refined
    by gradient ascent of the overlap (a few mean-shift steps), so relaxed
    planes find their best alignment too.  Among equally good alignments the
    identity operation is kept.
    """
    if len(A) != len(B) or sorted(A[:, 0]) != sorted(B[:, 0]):
        return 0.0, None, None
    cell2d = np.asarray(cell2d, dtype=float)
    images = _wave_images(cell2d) if images is None else images
    ops = [np.eye(2, dtype=int)] if ops is None else ops
    species, counts = np.unique(A[:, 0], return_counts=True)
    anchor = A[A[:, 0] == species[np.argmin(counts)]][0]
    targets = B[B[:, 0] == anchor[0]]
    best = (-1.0, None, None)
    for W in ops:
        fa = A[:, 1:3] @ np.asarray(W, dtype=float)
        for target in targets:
            t = target[1:3] - anchor[1:3] @ np.asarray(W, dtype=float)
            for _ in range(5):  # mean shift: move t along the overlap gradient
                num, den = np.zeros(2), 0.0
                for Z in species:
                    ia, ib = A[:, 0] == Z, B[:, 0] == Z
                    d = B[ib][None, :, 1:3] - (fa[ia] + t)[:, None, :]
                    d -= np.round(d)
                    dd = d[:, :, None, :] - images[None, None, :, :]
                    cart = dd @ cell2d
                    dz = (B[ib][None, :, 3] - A[ia][:, None, 3])[:, :, None]
                    w = np.exp(-(np.sum(cart ** 2, axis=-1) + dz ** 2) / (4.0 * _WAVE_SIGMA ** 2))
                    num += np.einsum("ijk,ijkl->l", w, dd)
                    den += w.sum()
                step = num / den if den > 0 else np.zeros(2)
                t = t + step
                if np.linalg.norm(step @ cell2d) < 1e-6:
                    break
            s = _wave_similarity(A, B, cell2d, t % 1.0, W, images)
            if s > best[0] + 1e-9:
                best = (s, np.asarray(W), t % 1.0)
    return best


def _wave_shifts(A, B, cell2d, images=None, same_plane=_SAME_PLANE):
    """
    Every in-plane shift *t* (fractional) that moves plane *A* onto plane *B*
    without rotation (overlap at least *same_plane*): one for a plane with no
    translational symmetry inside the cell, several otherwise.
    """
    if len(A) != len(B) or sorted(A[:, 0]) != sorted(B[:, 0]):
        return []
    cell2d = np.asarray(cell2d, dtype=float)
    images = _wave_images(cell2d) if images is None else images
    species, counts = np.unique(A[:, 0], return_counts=True)
    anchor = A[A[:, 0] == species[np.argmin(counts)]][0]
    found = []
    for target in B[B[:, 0] == anchor[0]]:
        t = (target[1:3] - anchor[1:3]) % 1.0
        if _wave_similarity(A, B, cell2d, t, images=images) < same_plane:
            continue
        if not any(_frac_distance(t, u, cell2d) < 1e-3 for u in found):
            found.append(t)
    return found


def assign_plane_names(planes_sorted, atoms=None, axis=2, same_plane=_SAME_PLANE):
    """
    Label planes by their arrangement and their stacking phase.

    Each plane is treated as a wave: a smooth periodic density with one
    Gaussian per atom (:func:`_wave_overlap`).  Two planes have the same
    **arrangement** when one wave overlaps the other after some rotation or
    mirror of the in-plane lattice and some shift; they are in the same
    **phase** when they overlap as they are.  The label is the composition
    (metals first, e.g. ``O4``, ``Ir2O2``), a letter when one composition has
    several arrangements (``IrO2-a``, ``IrO2-b``), and one prime per phase
    after the first: ``O``, ``O'``, ``O''`` are the same arrangement shifted
    or rotated, i.e. different stackings.  Phases are numbered in stacking
    order (the order of *planes_sorted*, bottom to top of the bulk cell);
    letters follow a rotation- and translation-invariant fingerprint of the
    arrangement, so they do not depend on the bulk origin.

    Reconstruction (``-recon``) and deformation (``~``) suffixes are added by
    callers, not here.  Without *atoms*, planes are labelled by composition
    only.

    Parameters
    ----------
    planes_sorted : list of dict
        Planes from :func:`identify_planes`, in stacking order.
    atoms : Atoms or None
        Structure the plane indices refer to.
    axis : int
        Stacking axis.
    same_plane : float
        Normalised overlap (0-1) from which two waves count as the same.

    Returns
    -------
    names : list of str
        ``names[i]`` is the label of ``planes_sorted[i]``.
    name_map : dict
        ``{label: counts_dict}``.
    """
    import string

    formulas = [_formula_label(p["counts"]) for p in planes_sorted]
    if atoms is None:
        return formulas, {f: dict(p["counts"]) for f, p in zip(formulas, planes_sorted)}

    waves, cell2d = _plane_waves(atoms, planes_sorted, axis)
    ops = _lattice_point_ops(cell2d)
    images = _wave_images(cell2d)

    # Arrangements (any rotation and shift), then phases (as they are).
    arrangements = []          # [formula, reference plane, [phase reference planes]]
    member = []                # (arrangement index, phase index) per plane
    for i, wave in enumerate(waves):
        for a, (formula, ref, phases) in enumerate(arrangements):
            if formula != formulas[i]:
                continue
            if _wave_alignment(waves[ref], wave, cell2d, ops, images)[0] < same_plane:
                continue
            for k, p in enumerate(phases):
                if _wave_similarity(waves[p], wave, cell2d, images=images) >= same_plane:
                    member.append((a, k))
                    break
            else:
                phases.append(i)
                member.append((a, len(phases) - 1))
            break
        else:
            arrangements.append([formulas[i], i, [i]])
            member.append((len(arrangements) - 1, 0))

    # Letters for compositions with several arrangements, ordered by a
    # fingerprint that does not change under the lattice point operations.
    letter = {}
    by_formula = {}
    for a, (formula, ref, _) in enumerate(arrangements):
        by_formula.setdefault(formula, []).append(a)
    for formula, group in by_formula.items():
        if len(group) == 1:
            letter[group[0]] = ""
            continue
        if len(group) > len(string.ascii_lowercase):
            raise ValueError(f"Too many arrangements of {formula} planes (more than 26).")
        sigs = {}
        for a in group:
            ref = waves[arrangements[a][1]]
            images_of_ref = [
                [(row[0], *((row[1:3] @ np.asarray(W, dtype=float)) % 1.0)) for row in ref]
                for W in ops
            ]
            candidates = [_plane_signature(g) for g in images_of_ref]
            sigs[a] = sorted(candidates, key=cmp_to_key(_compare_signatures))[0]
        order = sorted(group, key=cmp_to_key(lambda i, j: _compare_signatures(sigs[i], sigs[j])))
        for ch, a in zip(string.ascii_lowercase, order):
            letter[a] = f"-{ch}"

    # Phases related by a rotation or mirror are ordered by a fingerprint
    # that sees rotations but not shifts, so their marks do not depend on
    # the bulk origin; phases that differ only by a shift keep stacking order.
    phase_rank = {}
    for a, (_, _, phases) in enumerate(arrangements):
        sigs = [_plane_signature([tuple(row[:3]) for row in waves[p]]) for p in phases]
        order = sorted(range(len(phases)),
                       key=cmp_to_key(lambda i, j: _compare_signatures(sigs[i], sigs[j])))
        for rank, k in enumerate(order):
            phase_rank[(a, k)] = rank
    names = [f"{arrangements[a][0]}{letter[a]}{_phase_suffix(phase_rank[(a, k)])}" for a, k in member]
    name_map = {}
    for name, plane in zip(names, planes_sorted):
        name_map.setdefault(name, dict(plane["counts"]))
    return names, name_map


def _plane_waves(atoms, planes_sorted, axis=2):
    """Waves (rows ``(Z, fx, fy, dz)``) of the planes and the in-plane cell."""
    ab_axes = [i for i in range(3) if i != axis]
    frac = atoms.get_scaled_positions(wrap=False)
    cell2d = np.array(atoms.cell)[np.ix_(ab_axes, ab_axes)]
    height = float(atoms.cell.lengths()[axis])
    z = atoms.positions[:, axis]
    waves = []
    for plane in planes_sorted:
        idx = np.asarray(plane["indices"], dtype=int)
        dz = z[idx] - (plane["z_center"] if "z_center" in plane else z[idx[0]])
        if height > 0:
            dz = ((dz + 0.5 * height) % height) - 0.5 * height
        if "z_center" not in plane:
            dz = dz - dz.mean()
        waves.append(np.column_stack([atoms.numbers[idx], frac[idx, ab_axes[0]] % 1.0,
                                      frac[idx, ab_axes[1]] % 1.0, dz]))
    return waves, cell2d


def _sandwich(waves, planes_sorted, i, below=None, above=None):
    """
    Wave of plane *i* with the planes directly below and above it, heights
    relative to plane *i*: the plane in its stacking context.  *below* and
    *above* override the neighbours as ``(wave, dz, shift)`` (e.g. a plane
    of the next repeat unit, *dz* its height above plane *i* and *shift* its
    fractional in-plane offset).
    """
    z = [p["z_center"] for p in planes_sorted]
    rows = [waves[i]]
    for k, given in ((i - 1, below), (i + 1, above)):
        if given is None:
            if not 0 <= k < len(waves):
                continue
            given = (waves[k], z[k] - z[i], (0.0, 0.0))
        wave, dz, shift = given
        moved = wave.copy()
        moved[:, 1:3] = (moved[:, 1:3] + np.asarray(shift, dtype=float)) % 1.0
        moved[:, 3] += dz
        rows.append(moved)
    return np.vstack(rows)


def _cell_sandwiches(waves, planes_sorted, L, a3_xy):
    """Sandwiches of the planes of a periodic bulk cell: the neighbours of the
    first and last planes come from the repeat units below and above."""
    n = len(planes_sorted)
    z = [p["z_center"] for p in planes_sorted]
    a3_xy = np.asarray(a3_xy, dtype=float)
    out = []
    for i in range(n):
        below = above = None
        if i == 0:
            below = (waves[n - 1], z[n - 1] - L - z[0], -a3_xy)
        if i == n - 1:
            above = (waves[0], z[0] + L - z[n - 1], a3_xy)
        if n == 1:
            below = (waves[0], -L, -a3_xy)
            above = (waves[0], L, a3_xy)
        out.append(_sandwich(waves, planes_sorted, i, below, above))
    return out


def _cell_repeat(planes_sorted, waves, cell2d, L, a3_xy, same_plane=_SAME_PLANE):
    """
    Smallest number of planes after which the stacking of a periodic bulk
    cell repeats by a lattice translation.

    Usually all planes of the cell (``len(planes_sorted)``); fewer when the
    cell holds lattice-equivalent copies, e.g. a body-centred bulk or a bulk
    supercell along the normal.  Plane ``i + per`` must be plane ``i`` moved
    by one in-plane shift together with its neighbours (:func:`_sandwich`),
    so relaxation noise in the spacings only lowers the overlap a little;
    past the top of the cell it is a plane of the next repeat unit, shifted
    in-plane by *a3_xy* (fractional).
    """
    n = len(planes_sorted)
    images = _wave_images(cell2d)
    a3_xy = np.asarray(a3_xy, dtype=float)
    sandwiches = _cell_sandwiches(waves, planes_sorted, L, a3_xy)
    for per in range(1, n):
        if n % per or planes_sorted[per]["counts"] != planes_sorted[0]["counts"]:
            continue
        # Plane `per` sits in this cell; its neighbourhood fixes the shift.
        s, _, t = _wave_alignment(sandwiches[0], sandwiches[per], cell2d, images=images)
        if s < same_plane:
            continue
        for i in range(n):
            j, m = (i + per) % n, (i + per) // n
            if planes_sorted[j]["counts"] != planes_sorted[i]["counts"]:
                break
            if _wave_similarity(sandwiches[i], sandwiches[j], cell2d, t - m * a3_xy,
                                images=images) < same_plane:
                break
        else:
            return per
    return n


def _repeat_names(planes_sorted, atoms, L, a3_xy, axis=2):
    """
    Labels of the planes of one bulk cell (:func:`assign_plane_names`), with
    lattice-equivalent copies inside the cell sharing a label: their
    relative phase is the same.  Returns ``(names, name_map, per)`` with
    *per* the number of planes in one lattice repeat.
    """
    waves, cell2d = _plane_waves(atoms, planes_sorted, axis)
    per = _cell_repeat(planes_sorted, waves, cell2d, L, a3_xy)
    first, _ = assign_plane_names(planes_sorted[:per], atoms=atoms, axis=axis)
    names = [first[i % per] for i in range(len(planes_sorted))]
    name_map = {}
    for name, plane in zip(names, planes_sorted):
        name_map.setdefault(name, dict(plane["counts"]))
    return names, name_map, per


def _surface_a3_xy(bulk_atoms, miller, cell2d):
    """In-plane part of the stacking vector, fractional in *cell2d*."""
    a3 = np.asarray(surface_bulk_cell(bulk_atoms, miller)[2], dtype=float)
    return np.linalg.solve(np.asarray(cell2d, dtype=float).T, a3[:2])


def _slab_repeat(planes_sorted, waves, cell2d, same_plane=_SAME_PLANE):
    """
    The lattice repeat of a slab, learned from its most bulk-like region:
    the smallest ``per`` such that, over a run of at least ``per``
    consecutive planes ``i``, plane ``i + per`` with its neighbours is plane
    ``i`` with its neighbours moved by an in-plane shift (:func:`_sandwich`).  Relaxed surfaces, or a distorted middle, do not
    have to match.  Returns ``(per, t, dz, first)``: *dz* the mean height of
    one repeat over the run and *first* the run's first plane; or ``None``
    when no region of the slab shows one repeat unit twice.
    """
    n = len(planes_sorted)
    images = _wave_images(cell2d)
    z = np.array([p["z_center"] for p in planes_sorted])
    sandwiches = [_sandwich(waves, planes_sorted, i) for i in range(n)]
    for per in range(1, n):
        idx = list(range(1, n - 1 - per))  # both planes have both neighbours
        if len(idx) < per:
            return None
        shifts = []
        for i in idx:
            t = None
            if planes_sorted[i + per]["counts"] == planes_sorted[i]["counts"]:
                # Align the planes with their neighbours: a symmetric plane maps
                # onto itself by several shifts, only the lattice translation
                # also maps the neighbours.
                s, _, t_i = _wave_alignment(sandwiches[i], sandwiches[i + per], cell2d, images=images)
                if s >= same_plane:
                    t = t_i
            shifts.append(t)
        # Longest run of consecutive matches.  Their shifts may differ by a
        # translation of the in-plane cell (when it is a supercell of the
        # primitive one); any of them moves a plane onto its copy.
        best, run = [], []
        for i, t in zip(idx, shifts):
            run = run + [i] if t is not None else []
            if len(run) > len(best):
                best = list(run)
        if len(best) >= per:
            t = shifts[idx.index(best[0])]
            dz = float(np.mean([z[i + per] - z[i] for i in best]))
            return per, t, dz, best[0]
    return None


def _slab_plane_names(atoms, planes_sorted, axis=2, stacking_labels=None):
    """
    Labels of the planes of a slab, by relative phase: copies one lattice
    repeat apart share a label, as in the bulk cell.

    The repeat is learned from the slab's interior (:func:`_slab_repeat`).
    One interior repeat unit is labelled -- with *stacking_labels* (one
    repeat unit of genslab's labels, starting at the slab's bottom plane)
    when they fit, so genslab and cutslab name planes alike, else by
    :func:`assign_plane_names` starting after the widest gap as genslab does
    -- and every plane gets the label of the reference plane it is a copy
    of: the same composition, the same wave once moved by the in-plane
    shift of the repeats between them, and the nearest height modulo the
    repeat.  A plane that is a copy of none (a relaxed surface plane) gets
    the label of the reference plane with its composition nearest in
    height, with ``~``.  Returns ``(names, repeat)``; *repeat* is ``None``
    when the slab is too thin, and planes are then labelled one by one
    (absolute phases).
    """
    waves, cell2d = _plane_waves(atoms, planes_sorted, axis)
    n = len(planes_sorted)
    found = _slab_repeat(planes_sorted, waves, cell2d) if n >= 3 else None
    if found is None and stacking_labels:
        # Too thin to show its repeat twice, but genslab told us one repeat
        # unit of labels from the bottom plane up: use them if every plane
        # has the composition its label says (the two surface planes may
        # differ: reconstructed or relaxed).
        per = len(stacking_labels)
        cyclic = [stacking_labels[i % per] for i in range(n)]
        if all(plane_name_base(cyclic[i]) == _formula_label(planes_sorted[i]["counts"])
               for i in range(1, n - 1)):
            names = list(cyclic)
            for i in {0, n - 1}:
                if plane_name_base(names[i]) != _formula_label(planes_sorted[i]["counts"]):
                    names[i] = _formula_label(planes_sorted[i]["counts"]) + "~"
            return names, (per, None, None)
    if found is None:
        names, _ = assign_plane_names(planes_sorted, atoms=atoms, axis=axis)
        return names, None
    per, t, dz_rep, first = found
    images = _wave_images(cell2d)
    z = np.array([p["z_center"] for p in planes_sorted])

    # Reference repeat unit in the bulk-like region the repeat was learned
    # from, starting after the widest gap between planes (as genslab's cell).
    ref = list(range(first, first + per))
    gaps = [(z[ref[(k + 1) % per]] + (dz_rep if k == per - 1 else 0.0)) - z[ref[k]]
            for k in range(per)]
    start = (int(np.argmax(np.round(gaps, 3))) + 1) % per
    # One consecutive repeat unit from there: the planes before the window
    # are replaced by their copies one repeat up (same crystal planes, but
    # moved in-plane by the repeat translation, so their phases differ).
    ref = ref[start:] + [r + per for r in ref[:start]]

    def residual(i, r):
        """Height of plane i above reference plane r, modulo the repeat."""
        return ((z[i] - z[r] + 0.5 * dz_rep) % dz_rep) - 0.5 * dz_rep

    def copy_of(i):
        """The reference class plane i is a copy of, or None."""
        best = None
        for c, r in enumerate(ref):
            if planes_sorted[i]["counts"] != planes_sorted[r]["counts"]:
                continue
            k = int(np.round((z[i] - z[r]) / dz_rep))
            moved = waves[r].copy()
            moved[:, 1:3] = (moved[:, 1:3] + k * np.asarray(t)) % 1.0
            if _wave_similarity(moved, waves[i], cell2d, images=images) < _SAME_PLANE:
                continue
            if best is None or abs(residual(i, r)) < abs(residual(i, ref[best])):
                best = c
        return best

    classes = [copy_of(i) for i in range(n)]
    ref_names = None
    if stacking_labels is not None and len(stacking_labels) == per:
        # Align genslab's labels: stacking_labels[k] is the slab's k-th plane
        # from the bottom.  The bottom plane may be relaxed or reconstructed,
        # so the plane above it can anchor the alignment instead.
        for i in range(min(2, n)):
            if classes[i] is not None:
                ref_names = [stacking_labels[(i + c - classes[i]) % per] for c in range(per)]
                break
    if ref_names is None:
        ref_names, _ = assign_plane_names([planes_sorted[r] for r in ref], atoms=atoms, axis=axis)

    names = []
    for i, c in enumerate(classes):
        if c is not None:
            names.append(ref_names[c])
            continue
        same = [c for c, r in enumerate(ref) if planes_sorted[r]["counts"] == planes_sorted[i]["counts"]]
        if same:
            c = min(same, key=lambda c: abs(residual(i, ref[c])))
            names.append(_undeformed(ref_names[c]) + "~")
        else:
            names.append(_formula_label(planes_sorted[i]["counts"]) + "~")
    return names, (per, t, dz_rep)


def _gauss_reduce_basis(cell2d):
    """Integer unimodular ``P`` such that ``P @ cell2d`` is a Lagrange-Gauss reduced basis."""
    P = np.eye(2, dtype=int)
    B = np.array(cell2d, dtype=float)
    for _ in range(100):
        if np.dot(B[1], B[1]) < np.dot(B[0], B[0]):
            B = B[::-1].copy()
            P = P[::-1].copy()
        mu = int(np.round(np.dot(B[0], B[1]) / np.dot(B[0], B[0])))
        if mu == 0:
            break
        B[1] -= mu * B[0]
        P[1] -= mu * P[0]
    return P


def _lattice_point_ops(cell2d, tol=1e-3):
    """
    Point-group operations of the 2D lattice with basis rows *cell2d*, as
    integer matrices ``W`` with ``W @ cell2d`` the rotated basis: 8 for a
    square lattice, 12 hexagonal, 4 (centred) rectangular, 2 oblique.
    """
    P = _gauss_reduce_basis(cell2d)
    P_inv = np.round(np.linalg.inv(P)).astype(int)
    reduced = P @ np.asarray(cell2d, dtype=float)
    G = reduced @ reduced.T
    atol = tol * float(np.max(np.abs(G)))
    ops = []
    for entries in product((-1, 0, 1), repeat=4):
        W = np.array(entries, dtype=int).reshape(2, 2)
        if abs(round(np.linalg.det(W))) != 1:
            continue
        if np.allclose(W @ G @ W.T, G, atol=atol):
            ops.append(P_inv @ W @ P)
    return ops


def _formula_label(counts):
    """Composition label of a plane, e.g. ``{8: 4} -> 'O4'``, ``{77: 1, 8: 2} -> 'IrO2'``."""
    return Formula.from_dict(
        {chemical_symbols[int(Z)]: int(c) for Z, c in counts.items() if c > 0}
    ).format("metal")


# In-plane reciprocal vectors (h, k) of the plane fingerprint, and pairs (a, b)
# of its triplet invariants F(a) F(b) F(-a-b).
_SIG_G = [(1, 0), (0, 1), (1, 1), (1, -1), (2, 0), (0, 2), (2, 1), (1, 2), (2, -1), (1, -2)]
_SIG_TRIPLETS = [((1, 0), (0, 1)), ((1, 0), (1, 0)), ((0, 1), (0, 1)), ((1, 0), (0, -1)),
                 ((1, 1), (1, -1)), ((2, 0), (-1, 1)), ((0, 2), (1, -1))]


def _plane_signature(geom):
    """
    Translation-invariant fingerprint of a plane ``[(Z, fx, fy), ...]``.

    Magnitudes of the Z-weighted structure factor ``F(g)`` over
    :data:`_SIG_G`, then the imaginary parts of triplet invariants
    ``F(a) F(b) F(-a-b)``, which tell a plane from its 180-degree rotation.
    ``F`` is normalised by the total weight, so every entry is in [-1, 1]
    and changes smoothly with the atom positions.
    """
    w = np.array([a[0] for a in geom], dtype=float)
    f = np.array([[a[1], a[2]] for a in geom], dtype=float)

    def F(g):
        return np.sum(w * np.exp(2j * np.pi * (f @ np.asarray(g, dtype=float)))) / w.sum()

    mags = [abs(F(g)) for g in _SIG_G]
    trips = [(F(a) * F(b) * F(-np.add(a, b))).imag for a, b in _SIG_TRIPLETS]
    return np.array(mags + trips)


def _compare_signatures(a, b, tol=0.02):
    """
    Order two plane fingerprints, robustly to small displacements.

    Equal if no entry differs by more than *tol*; otherwise decided by the
    first entry whose difference is at least half the largest one (mirror
    variants have swapped entries of equal size, so the first one decides).
    """
    d = np.asarray(a) - np.asarray(b)
    big = float(np.max(np.abs(d)))
    if big <= tol:
        return 0
    k = int(np.flatnonzero(np.abs(d) >= 0.5 * big)[0])
    return -1 if d[k] < 0 else 1


#: How a plane label selects planes (``prefer_plane``, ``cut_at``):
#: ``"shape"`` -- the arrangement in any phase (any shift or rotation);
#: ``"relative"`` -- only that plane: same arrangement, same phase in the
#: crystal (copies a whole repeat unit apart are the same plane);
#: ``"absolute"`` -- also in phase with the reference surface in the slab.
SELECTIONS = ("shape", "relative", "absolute")

_LABEL_RE = re.compile(r"^(?P<body>.*?)(?P<phase>'\d+|'*)$")


def _phase_suffix(k):
    """Phase mark of the *k*-th phase: none, then one to three primes, then
    a prime and the number (``'4``, ``'5``, ...)."""
    if k <= 0:
        return ""
    return "'" * k if k <= 3 else f"'{k}"


def _parse_plane_name(name):
    """
    Split a plane label into ``(composition, letter, phase, recon, deformed)``,
    e.g. ``IrO2-a`` with two primes, ``-recon`` and ``~`` gives
    ``("IrO2", "a", 2, True, True)``, and ``O'5`` gives
    ``("O", None, 5, False, False)``.
    """
    core = name or ""
    deformed = core.endswith("~")
    if deformed:
        core = core[:-1]
    recon = core.endswith("-recon")
    if recon:
        core = core[:-6]
    m = _LABEL_RE.match(core)
    body, mark = m.group("body"), m.group("phase")
    phase = int(mark[1:]) if mark[1:].isdigit() else len(mark)
    head, sep, tail = body.rpartition("-")
    if sep and len(tail) == 1 and tail.isalpha() and tail.islower():
        return head, tail, phase, recon, deformed
    return body, None, phase, recon, deformed


def _undeformed(name):
    """The label without the deformation mark ``~``."""
    return name[:-1] if name and name.endswith("~") else name


def plane_name_base(name):
    """
    Return the composition part of a plane label.

    Examples: ``IrO2-a`` gives ``IrO2``, ``O4'-recon`` gives ``O4``,
    ``IrO2-b''~`` gives ``IrO2``.
    """
    if not name:
        return name
    return _parse_plane_name(name)[0]


def plane_name_for_filename(name):
    """
    A plane label safe in file names: each prime becomes ``p`` and the
    deformation mark ``~`` becomes ``d`` (``O4'`` -> ``O4p``,
    ``IrO2-a''-recon`` -> ``IrO2-app-recon``, ``O4~`` -> ``O4d``).
    """
    return name.replace("'", "p").replace("~", "d")


def plane_name_matches(query, name, selection="relative"):
    """
    Whether *query* selects plane label *name*.

    A label is ``composition[-letter][phase][-recon][~]``, e.g. ``O4``,
    ``IrO2-a'``, ``O4''-recon``, ``O4~`` (deformed).  The query must give
    the same composition, and its letter if the name has one.  *selection*
    decides about the phase (see :data:`SELECTIONS`):

    - ``"shape"``: any phase (``O`` selects ``O``, ``O'``, ``O''``; a bare
      composition selects every arrangement, ``IrO2`` selects ``IrO2-a`` and
      ``IrO2-b``)
    - ``"relative"`` / ``"absolute"``: the same arrangement and phase
      (``O'`` selects ``O'`` only); the absolute phase is checked on the
      geometry by the caller, not here

    In every mode a query without ``-recon`` also selects the reconstructed
    plane (``O4`` selects ``O4-recon``), and a query without ``~`` selects
    the deformed plane (``O4`` selects ``O4~``).
    """
    if selection not in SELECTIONS:
        raise ValueError(f"selection must be one of {SELECTIONS}, got {selection!r}")
    q_comp, q_letter, q_phase, q_recon, q_deformed = _parse_plane_name(query)
    n_comp, n_letter, n_phase, n_recon, n_deformed = _parse_plane_name(name)
    if q_comp != n_comp:
        return False
    if (q_recon and not n_recon) or (q_deformed and not n_deformed):
        return False
    if selection == "shape":
        return q_letter is None or q_letter == n_letter
    return q_letter == n_letter and q_phase == n_phase


def _plane_translations(ref, tgt, cell2d, tol):
    """
    All in-plane translations mapping plane *ref* onto plane *tgt*.

    *ref* and *tgt* are lists of ``(Z, fx, fy)`` in fractional coordinates
    of the in-plane lattice with basis rows *cell2d*.  Returns the distinct
    fractional translations ``t`` (in ``[0, 1)``) such that every ref atom
    moved by ``t`` lies within *tol* angstrom of a distinct tgt atom of the
    same species.  Several are found when a plane maps onto itself under a
    shift that is not a lattice vector.
    """
    pairing = _plane_pairing(ref, tgt)
    if pairing is None:
        return []
    if len(ref) == 0:
        return [np.zeros(2)]
    found = []
    for t in pairing["shifts"]:
        t = t % 1.0
        if _shift_matches(ref, tgt, t, cell2d, tol) and not any(
            _frac_distance(t, u, cell2d) < 1e-3 for u in found
        ):
            found.append(t)
    return found


def _plane_pairing(ref, tgt):
    """
    Common set-up for matching plane *ref* onto plane *tgt* (atoms
    ``(Z, fx, fy, ...)``): ``None`` if their compositions differ, else the
    fractional positions, the atom indices of each species in both planes,
    and the candidate shifts that put one atom of the rarest species of
    *ref* onto each atom of that species in *tgt*.
    """
    if len(ref) != len(tgt):
        return None
    ref_Z = np.array([a[0] for a in ref], dtype=int)
    tgt_Z = np.array([a[0] for a in tgt], dtype=int)
    species, counts = np.unique(ref_Z, return_counts=True)
    tgt_species, tgt_counts = np.unique(tgt_Z, return_counts=True)
    if not (np.array_equal(species, tgt_species) and np.array_equal(counts, tgt_counts)):
        return None
    ref_f = np.array([[a[1], a[2]] for a in ref], dtype=float).reshape(-1, 2)
    tgt_f = np.array([[a[1], a[2]] for a in tgt], dtype=float).reshape(-1, 2)
    shifts = []
    if len(ref):
        anchor_Z = species[np.argmin(counts)]
        anchor = int(np.flatnonzero(ref_Z == anchor_Z)[0])
        shifts = [tgt_f[b] - ref_f[anchor] for b in np.flatnonzero(tgt_Z == anchor_Z)]
    return {
        "ref_f": ref_f,
        "tgt_f": tgt_f,
        "groups": [(np.flatnonzero(ref_Z == Z), np.flatnonzero(tgt_Z == Z)) for Z in species],
        "shifts": shifts,
    }


def _frac_distance(a, b, cell2d):
    """Minimum-image distance (angstrom) between fractional in-plane points."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d -= np.round(d)
    return float(np.linalg.norm(d @ np.asarray(cell2d, dtype=float)))


def _shift_matches(ref, tgt, t, cell2d, tol):
    """Whether every atom of *ref* moved by *t* lies within *tol* angstrom of
    a distinct atom of *tgt* of the same species (one-to-one)."""
    if len(ref) != len(tgt):
        return False
    ref_Z = np.array([a[0] for a in ref], dtype=int)
    tgt_Z = np.array([a[0] for a in tgt], dtype=int)
    ref_f = np.array([[a[1], a[2]] for a in ref], dtype=float).reshape(-1, 2)
    tgt_f = np.array([[a[1], a[2]] for a in tgt], dtype=float).reshape(-1, 2)
    for Z in np.unique(ref_Z):
        r_idx, t_idx = np.flatnonzero(ref_Z == Z), np.flatnonzero(tgt_Z == Z)
        if len(r_idx) != len(t_idx):
            return False
        d = (ref_f[r_idx] + t)[:, None, :] - tgt_f[t_idx][None, :, :]
        d -= np.round(d)
        dist = np.linalg.norm(d @ np.asarray(cell2d, dtype=float), axis=-1)
        rows, cols = linear_sum_assignment(dist)
        if dist[rows, cols].max() > tol:
            return False
    return True


def _bulk_plane_catalog(bulk_atoms, miller, plane_tol=None):
    """
    Planes of one bulk repeat unit in the frame of :func:`build_surface`.

    Returns ``(catalog, L, cell2d)`` where each catalog entry has the
    plane ``label`` (as genslab names it), ``counts``, ``z`` (centre, mod L)
    and ``atoms``: ``(Z, fx, fy, dz)`` with *dz* the height relative to the
    plane centre.
    """
    surf = build_surface(bulk_atoms, miller, layers=1)
    L = float(surf.cell[2, 2])
    z = surf.positions[:, 2]
    atoms_z = np.column_stack([surf.numbers, z, np.zeros(len(surf))])
    planes = sorted(identify_planes(atoms_z, L, plane_tol=plane_tol), key=lambda p: p["z_center"])
    labels, _, _ = _repeat_names(planes, surf, L, _surface_a3_xy(bulk_atoms, miller, surf.cell[:2, :2]))
    frac = surf.get_scaled_positions()
    catalog = []
    for plane, label in zip(planes, labels):
        zc = plane["z_center"]
        catalog.append({
            "label": label,
            "counts": dict(plane["counts"]),
            "z": zc,
            "atoms": [
                (int(surf.numbers[i]), frac[i, 0], frac[i, 1],
                 ((z[i] - zc + 0.5 * L) % L) - 0.5 * L)
                for i in plane["indices"]
            ],
        })
    return catalog, L, np.array(surf.cell[:2, :2])


def _supercell_matrix(bulk_cell2d, cell2d, atol=0.02):
    """Integer matrix ``M`` with ``cell2d = M @ bulk_cell2d`` (in-plane rows),
    allowing a strain of about *atol* between the two cells."""
    M = np.asarray(cell2d, dtype=float) @ np.linalg.inv(bulk_cell2d)
    M_int = np.round(M).astype(int)
    det = int(round(np.linalg.det(M_int)))
    if not np.allclose(M, M_int, atol=atol) or det == 0:
        raise ValueError(
            "The slab's in-plane cell is not an integer supercell of the bulk "
            "surface cell; check bulk_atoms and miller."
        )
    return M_int, det


def _supercell_translations(M_int, det):
    """The ``|det|`` bulk lattice translations (integer pairs) inside the
    supercell ``M_int``, found exactly with the adjugate of ``M_int``."""
    adj = np.array([[M_int[1, 1], -M_int[0, 1]], [-M_int[1, 0], M_int[0, 0]]])
    corners = np.array([[0, 0], M_int[0], M_int[1], M_int[0] + M_int[1]])
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    out = []
    for i in range(lo[0], hi[0] + 1):
        for j in range(lo[1], hi[1] + 1):
            v = np.array([i, j]) @ adj  # = det * (fractional position in the supercell)
            if (det > 0 and np.all((v >= 0) & (v < det))) or (
                det < 0 and np.all((v <= 0) & (v > det))
            ):
                out.append((i, j))
    return out


def _tile_plane(plane_atoms, bulk_cell2d, cell2d):
    """Express plane atoms ``(Z, fx, fy, ...)`` of in-plane cell *bulk_cell2d*
    in the integer in-plane supercell *cell2d*."""
    M_int, det = _supercell_matrix(bulk_cell2d, cell2d)
    if det == 1 and np.array_equal(M_int, np.eye(2, dtype=int)):
        return list(plane_atoms)
    M_inv = np.linalg.inv(M_int)
    tiled = []
    for t in _supercell_translations(M_int, det):
        for atom in plane_atoms:
            f = ((np.array(atom[1:3]) + t) @ M_inv) % 1.0
            tiled.append((atom[0], f[0], f[1]) + tuple(atom[3:]))
    return tiled


def _plane_rmsd(ref, tgt, cell2d):
    """
    RMSD (angstrom) between planes ``[(Z, fx, fy, dz), ...]`` after the best
    rigid in-plane shift (and removal of the mean height); ``inf`` if the
    compositions differ.
    """
    pairing = _plane_pairing(ref, tgt)
    if pairing is None:
        return float("inf")
    ref_f, tgt_f, groups = pairing["ref_f"], pairing["tgt_f"], pairing["groups"]
    ref_dz = np.array([a[3] for a in ref], dtype=float)
    tgt_dz = np.array([a[3] for a in tgt], dtype=float)
    ref_dz, tgt_dz = ref_dz - ref_dz.mean(), tgt_dz - tgt_dz.mean()
    cell2d = np.asarray(cell2d, dtype=float)

    def residuals(t):
        res = []
        for r_idx, t_idx in groups:
            d = tgt_f[t_idx][None, :, :] - (ref_f[r_idx] + t)[:, None, :]
            d -= np.round(d)
            cost = np.linalg.norm(d @ cell2d, axis=-1) ** 2
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                res.append((d[r, c], tgt_dz[t_idx[c]] - ref_dz[r_idx[r]]))
        return res

    best = float("inf")
    for t in pairing["shifts"]:
        for _ in range(2):  # refine the shift by the mean residual
            res = residuals(t)
            t = t + np.mean([d for d, _ in res], axis=0)
        res = residuals(t)
        msd = np.mean([np.sum((d @ cell2d) ** 2) + dz ** 2 for d, dz in res])
        best = min(best, float(np.sqrt(msd)))
    return best


def _catalog_in_cell(catalog, bulk_cell2d, cell2d, L, site_tol=0.3, a3_xy=(0.0, 0.0)):
    """
    Express the bulk plane catalog in the slab's in-plane cell *cell2d*, in
    place: tiled when the slab cell is a supercell of the bulk surface cell,
    folded (and relabelled) when *bulk_atoms* is a supercell of the cell the
    slab was built from.  A few per cent of strain between the cells is
    accepted.
    """
    M = np.asarray(cell2d, dtype=float) @ np.linalg.inv(bulk_cell2d)
    if np.allclose(M, np.round(M), atol=0.02) and round(np.linalg.det(np.round(M))) != 0:
        for entry in catalog:
            entry["atoms"] = _tile_plane(entry["atoms"], bulk_cell2d, cell2d)
            entry["counts"] = dict(Counter(a[0] for a in entry["atoms"]))
        return
    K = np.linalg.inv(M)  # bulk surface cell = K @ slab cell
    K_int = np.round(K).astype(int)
    copies = abs(int(round(np.linalg.det(K_int))))
    if not np.allclose(K, K_int, atol=0.02) or copies == 0:
        raise ValueError(
            "The slab's in-plane cell is not an integer supercell of the bulk surface "
            "cell, nor the other way round; check bulk_atoms and miller."
        )
    for entry in catalog:
        sites = []
        for Z, fx, fy, dz in entry["atoms"]:
            f = (np.array([fx, fy]) @ K_int) % 1.0
            for site in sites:
                if site[0][0] == Z and _frac_distance(site[0][1:3], f, cell2d) < site_tol:
                    site[1] += 1
                    break
            else:
                sites.append([(Z, f[0], f[1], dz), 1])
        if any(c != copies for _, c in sites):
            raise ValueError(
                "bulk_atoms is not periodic with the slab's in-plane cell; pass the bulk "
                "cell the slab was built from."
            )
        entry["atoms"] = [a for a, _ in sites]
        entry["counts"] = dict(Counter(a[0] for a in entry["atoms"]))
    # Labels count atoms per cell: recompute them in the slab's cell.
    from ase import Atoms

    numbers, scaled, planes, start = [], [], [], 0
    for entry in catalog:
        for Z, fx, fy, dz in entry["atoms"]:
            numbers.append(Z)
            scaled.append([fx, fy, ((entry["z"] + dz) / L) % 1.0])
        planes.append({"indices": list(range(start, len(numbers))), "counts": entry["counts"],
                       "z_center": entry["z"] % L})
        start = len(numbers)
    cell = np.zeros((3, 3))
    cell[:2, :2] = cell2d
    cell[2, 2] = L
    folded = Atoms(numbers=numbers, scaled_positions=scaled, cell=cell, pbc=True)
    labels, _, _ = _repeat_names(planes, folded, L, a3_xy)
    for entry, label in zip(catalog, labels):
        entry["label"] = label


class _BulkSites:
    """
    The bulk crystal as sites in the slab frame: the atoms of the plane
    catalog (in the slab's in-plane cell), repeated along the stacking vector
    (in-plane part *a3_xy*, fractional; height *L*).  A slab atom ``(f, z)``
    sits on site ``s`` of repeat ``m`` under the registry ``(t, offset,
    scale)`` when ``f = f_s + m * a3_xy + t`` (mod 1) and ``z = offset +
    scale * (z_s + m * L)``.
    """

    def __init__(self, catalog, L, cell2d, a3_xy, bulk_cell2d=None):
        self.L, self.cell2d = float(L), np.asarray(cell2d, dtype=float)
        self.a3_xy = np.asarray(a3_xy, dtype=float)
        # In-plane lattice of the sites: the bulk surface cell when the slab
        # cell is a supercell of it (shifts by its vectors are equivalent).
        self.lattice = self.cell2d
        if bulk_cell2d is not None:
            M = self.cell2d @ np.linalg.inv(bulk_cell2d)
            if np.allclose(M, np.round(M), atol=0.02) and abs(round(np.linalg.det(np.round(M)))) > 1:
                self.lattice = np.linalg.inv(np.round(M)) @ self.cell2d
        rows = [(Z, fx, fy, entry["z"] + dz, k)
                for k, entry in enumerate(catalog) for Z, fx, fy, dz in entry["atoms"]]
        rows = np.array(rows, dtype=float)
        self.Z, self.f, self.z, self.plane = rows[:, 0], rows[:, 1:3], rows[:, 3], rows[:, 4].astype(int)
        # Rounding the fractional difference finds the nearest image of every
        # pair closer than half the cell's smallest height; neighbouring
        # images are checked only when that is within reach of the densities.
        heights = abs(np.linalg.det(self.cell2d)) / np.linalg.norm(self.cell2d, axis=1)[::-1]
        reach = np.sqrt(-4.0 * _WAVE_SIGMA ** 2 * np.log(1e-12))
        self.images = (np.array([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)], dtype=float)
                       if heights.min() / 2 < reach else np.zeros((1, 2)))
        # Squared distance from every site to the nearest other site of its
        # element: the cost of one hop.
        own, _, _, df, dz = self.pairs(self.Z, self.f, self.z, (0.0, 0.0), 0.0, 1.0)
        d2 = np.sum((df @ self.cell2d) ** 2, axis=1) + dz ** 2
        d2[d2 < 1e-6] = np.inf
        self.hop2 = np.full(len(self.Z), np.inf)
        np.minimum.at(self.hop2, own, d2)

    def pairs(self, numbers, frac, z, t, offset, scale):
        """
        For every atom, its displacement from the same-element sites of the
        nearest repeats (nearest in-plane image): ``(atom, site, m, d_frac,
        dz)`` arrays, flattened over sites and repeats.
        """
        out = []
        for Z in np.unique(numbers):
            ia, js = np.flatnonzero(numbers == Z), np.flatnonzero(self.Z == Z)
            zb = (z[ia] - offset) / scale
            m0 = np.round((zb[:, None] - self.z[js][None, :]) / self.L)
            for dm in (-1, 0, 1):
                m = m0 + dm
                df = (frac[ia][:, None, :] - self.f[js][None, :, :]
                      - m[..., None] * self.a3_xy - np.asarray(t, dtype=float))
                df -= np.round(df)
                cand = df[:, :, None, :] + self.images[None, None, :, :]
                near = np.argmin(np.sum((cand @ self.cell2d) ** 2, axis=-1), axis=-1)
                df = np.take_along_axis(cand, near[:, :, None, None], axis=2)[:, :, 0, :]
                dz = z[ia][:, None] - offset - scale * (self.z[js][None, :] + m * self.L)
                shape = dz.shape
                out.append((np.broadcast_to(ia[:, None], shape).ravel(),
                            np.broadcast_to(js[None, :], shape).ravel(),
                            m.ravel(), df.reshape(-1, 2), dz.ravel()))
        return [np.concatenate(col) for col in zip(*out)]

    def reduce(self, t):
        """In-plane shift *t* (fractional in the slab cell) brought into one
        cell of the site lattice."""
        u = np.linalg.solve(self.lattice.T, np.asarray(t, dtype=float) @ self.cell2d) % 1.0
        return np.linalg.solve(self.cell2d.T, u @ self.lattice) % 1.0

    def weights(self, d_frac, dz, sigma):
        """Overlap of an atom and a site, both Gaussians of width *sigma*."""
        d2 = np.sum((d_frac @ self.cell2d) ** 2, axis=1) + dz ** 2
        return np.exp(-d2 / (4.0 * sigma ** 2))


def _bulk_registry(atoms, sites, sigma=_WAVE_SIGMA):
    """
    Register a slab on the bulk crystal: the in-plane shift *t*
    (fractional), height *offset* and strain *scale* along the normal that
    maximise the overlap of the slab's atoms with the bulk sites of their
    element (see :class:`_BulkSites`).  Every atom counts by how well it sits
    on a site, so bulk-like atoms decide and relaxed ones barely count; the
    in-plane positions tell apart planes only a fraction of an angstrom apart
    in height, and one shift for the whole slab keeps planes related by a
    glide or screw axis in their own phase.

    Registries start from putting an atom of the rarest element, taken at
    several heights, on each bulk site of that element, and are refined by
    mean-shift steps; the strain is then fitted on the best one, with the
    density sharpened step by step so that relaxed atoms drop out.  Among
    registries that fit equally well the one with the smallest shift is kept,
    so slabs cut by genslab keep its labels.  Returns ``(t, offset, scale)``.
    """
    numbers = atoms.numbers.astype(float)
    frac = atoms.get_scaled_positions(wrap=False)[:, :2]
    z = atoms.positions[:, 2]
    L, a3_xy, cell2d = sites.L, sites.a3_xy, sites.cell2d

    def normalise(t, offset):
        k = -np.floor(offset / L)
        return (np.asarray(t) + k * a3_xy) % 1.0, offset + k * L

    def refine(t, offset, scale, sigma, steps, fit_scale=False):
        for _ in range(steps):
            ia, js, m, df, dz = sites.pairs(numbers, frac, z, t, offset, scale)
            w = sites.weights(df, dz, sigma)
            if w.sum() <= 0:
                break
            step = (w[:, None] * df).sum(axis=0) / w.sum()
            t = t + step
            zb = sites.z[js] + m * L
            mean_zb = np.average(zb, weights=w)
            spread = np.sqrt(np.average((zb - mean_zb) ** 2, weights=w))
            if fit_scale and spread > 0.25 * L:
                previous = offset + scale * mean_zb
                scale, offset = (float(v) for v in np.polyfit(zb, z[ia], 1, w=np.sqrt(w)))
                dz_step = offset + scale * mean_zb - previous
            else:
                dz_step = np.average(dz, weights=w)
                offset = offset + dz_step
            if np.linalg.norm(step @ cell2d) < 1e-7 and abs(dz_step) < 1e-7:
                break
        return t, offset, scale

    def score(t, offset, scale):
        _, _, _, df, dz = sites.pairs(numbers, frac, z, t, offset, scale)
        return float(sites.weights(df, dz, sigma).sum())

    elements, counts = np.unique(numbers, return_counts=True)
    rare = elements[np.argmin(counts)]
    pool = np.flatnonzero(numbers == rare)
    pool = pool[np.argsort(z[pool])]
    seeds = sorted({int(pool[int(round(q * (len(pool) - 1)))]) for q in (0.1, 0.3, 0.5, 0.7, 0.9)})
    starts = []
    for i in seeds:
        for j in np.flatnonzero(sites.Z == rare):
            t, offset = normalise(frac[i] - sites.f[j], z[i] - sites.z[j])
            t = sites.reduce(t)
            if not any(abs(offset - o) < 0.05 and np.linalg.norm(((t - u + 0.5) % 1.0 - 0.5) @ cell2d) < 0.05
                       for u, o in starts):
                starts.append((t, offset))
    # Screen every start once; refine those within reach of the best.
    screened = [(score(t, offset, 1.0), t, offset) for t, offset in starts]
    top = max(sc for sc, _, _ in screened)
    best = None
    for s0, t, offset in sorted(screened, key=lambda x: -x[0]):
        if s0 < 0.5 * top:
            break
        t, offset, _ = refine(t, offset, 1.0, sigma, 50)
        t, offset = normalise(sites.reduce(t), offset)
        s = score(t, offset, 1.0)
        size = float(np.linalg.norm(((t + 0.5) % 1.0 - 0.5) @ cell2d))
        key = (s, -round(size, 3), -round(offset, 3))
        if best is None or s > best[0][0] * (1 + 1e-4) or (
                abs(s - best[0][0]) <= 1e-4 * best[0][0] and key[1:] > best[0][1:]):
            best = (key, t, offset)
    _, t, offset = best
    scale = 1.0
    for width in (sigma, 0.6 * sigma, 0.4 * sigma):
        t, offset, scale = refine(t, offset, scale, width, 20, fit_scale=True)
    return t % 1.0, offset, scale


def _assign_to_sites(numbers, frac, z, sites, t, offset, scale):
    """
    Assign every atom to its own bulk site under the registry, as a slab
    made of bulk planes: the contiguous run of bulk planes, and the
    assignment of atoms to its sites (one atom per site), that minimise the
    summed squared distance between atoms and their sites plus, for every
    site of the run left empty, the squared distance between neighbouring
    sites of its element (leaving a site empty costs as much as one hop).
    A relaxed surface atom that moved towards a site of the plane above thus
    stays in its own plane, and the half-occupied outer planes of a Tasker
    III slab are part of the run instead of the planes being shifted.

    Returns ``{(catalog plane, repeat m): [atom indices]}``.
    """
    L, cell2d = sites.L, sites.cell2d
    ia, js, m, df, dz = sites.pairs(numbers, frac, z, t, offset, scale)
    d2 = np.sum((df @ cell2d) ** 2, axis=1) + dz ** 2
    # Shortest distance of every atom to every (site, repeat) it may occupy.
    m = m.astype(int)
    span = int(m.max() - m.min()) + 1
    keys, inverse = np.unique(js * span + (m - m.min()), return_inverse=True)
    key_site, key_m = keys // span, keys % span + int(m.min())
    cost = np.full((len(numbers), len(keys)), np.inf)
    np.minimum.at(cost, (ia, inverse.ravel()), d2)
    plane_of = [(int(sites.plane[j]), int(mm)) for j, mm in zip(key_site, key_m)]
    centre = {k: sites.z[sites.plane == k].mean() for k in set(sites.plane.tolist())}
    height = {pl: offset + scale * (centre[pl[0]] + pl[1] * L) for pl in set(plane_of)}
    planes = sorted(height, key=height.get)
    index = {pl: n for n, pl in enumerate(planes)}
    site_plane = np.array([index[pl] for pl in plane_of])
    site_Z = sites.Z[key_site]
    vacancy = sites.hop2[key_site]
    need = Counter(numbers.tolist())
    elements = sorted(need)

    # Candidate runs: every start within a repeat of the slab's bottom, ends
    # from the first with room for every atom to one repeat beyond.
    runs = []
    for a, start_plane in enumerate(planes):
        if height[start_plane] > z.min() + L:
            break
        if height[start_plane] < z.min() - L:
            continue
        first = None
        for b in range(a, len(planes)):
            if first is not None and height[planes[b]] > height[planes[first]] + L:
                break
            in_run = (site_plane >= a) & (site_plane <= b)
            have = Counter(site_Z[in_run].tolist())
            if all(have[Z] >= n for Z, n in need.items()):
                first = b if first is None else first
                cols = np.flatnonzero(in_run)
                # Lower bound: every atom on its nearest site of the run, and
                # the cheapest sites left empty.
                vac = np.sort(vacancy[cols])
                bound = float(np.sum(np.min(cost[:, cols], axis=1))) + float(vac[:len(cols) - len(numbers)].sum())
                runs.append((bound, cols))

    best = None
    for bound, cols in sorted(runs, key=lambda r: r[0]):
        if best is not None and bound >= best[0]:
            break
        total, rows_all, picked_all, used = 0.0, [], [], np.zeros(len(cols), dtype=bool)
        for Z in elements:
            r = np.flatnonzero(numbers == Z)
            c = np.flatnonzero(site_Z[cols] == Z)
            sub = cost[np.ix_(r, cols[c])]
            if not np.isfinite(sub).any(axis=1).all():
                total = np.inf
                break
            rr, cc = linear_sum_assignment(np.where(np.isfinite(sub), sub, 1e12))
            total += float(sub[rr, cc].sum())
            used[c[cc]] = True
            rows_all.extend(r[rr])
            picked_all.extend(cols[c[cc]])
        if not np.isfinite(total):
            continue
        total += float(vacancy[cols][~used].sum())
        if best is None or total < best[0]:
            best = (total, rows_all, picked_all)
    groups = {}
    pairs = (zip(best[1], best[2]) if best is not None
             else enumerate(np.argmin(cost, axis=1)))  # no run fits: nearest sites
    for i, c in pairs:
        groups.setdefault(plane_of[c], []).append(int(i))
    return groups


def _planes_from_bulk(atoms, charges_list, bulk_atoms, miller, plane_tol=None,
                      charge_tol=1e-3, deform_tol=0.3):
    """
    Assign the atoms of a slab to the planes of its bulk and label them.

    The slab is registered on the bulk crystal (:func:`_bulk_registry`: one
    in-plane shift, a height and a strain along the normal, from how well its
    atoms sit on bulk sites of their element), and every atom goes to the
    plane of its nearest bulk site of that element, so rumpled or relaxed
    surface planes stay whole, even where bulk planes lie a fraction of an
    angstrom apart.  A plane gets the bulk label (e.g. ``O4``) when it matches
    its bulk plane within *deform_tol* (RMSD in angstrom after the best rigid
    shift) and a label with ``~`` (``O4~``) when it is more deformed or has a
    different composition.

    Returns ``(planes_sorted, labels, reduced_counts)`` with planes in the
    format of :func:`identify_planes`.
    """
    catalog, L, bulk_cell2d = _bulk_plane_catalog(bulk_atoms, miller, plane_tol)
    cell2d = np.array(atoms.cell[:2, :2])
    a3_xy = _surface_a3_xy(bulk_atoms, miller, cell2d)
    _catalog_in_cell(catalog, bulk_cell2d, cell2d, L, a3_xy=a3_xy)

    numbers = atoms.numbers
    z = atoms.positions[:, 2]
    q = np.asarray(charges_list, dtype=float)
    atoms_z = np.column_stack([numbers, z, q])
    missing = set(numbers.tolist()) - {Z for entry in catalog for Z in entry["counts"]}
    if missing:
        raise ValueError(
            f"The slab contains elements absent from the {tuple(miller)} planes of the "
            "bulk; check bulk_atoms and miller."
        )

    sites = _BulkSites(catalog, L, cell2d, a3_xy, bulk_cell2d)
    t, offset, scale = _bulk_registry(atoms, sites)

    frac = atoms.get_scaled_positions(wrap=False)
    groups = _assign_to_sites(numbers.astype(float), frac[:, :2], z, sites, t, offset, scale)

    def geometry(idx, zc):
        return [(int(numbers[i]), frac[i, 0] % 1.0, frac[i, 1] % 1.0, z[i] - zc) for i in idx]

    keyed = sorted(groups.items(), key=lambda kv: catalog[kv[0][0]]["z"] + kv[0][1] * L)
    planes_sorted, labels = [], []
    for (k, _), idx in keyed:
        idx = sorted(idx)
        plane = _make_plane(idx, z[idx], atoms_z, charge_tol)
        deviation = _plane_rmsd(catalog[k]["atoms"], geometry(idx, plane["z_center"]), cell2d)
        planes_sorted.append(plane)
        labels.append(catalog[k]["label"] + ("" if deviation <= deform_tol else "~"))

    bulk_numbers = np.asarray(bulk_atoms.numbers, dtype=float)
    reduced = compute_reduced_counts(np.column_stack([bulk_numbers, bulk_numbers * 0, bulk_numbers * 0]))
    return planes_sorted, labels, reduced


def compute_delete_info(cut_plane, deletion_mask, atoms_z_matrix, surf_bulk):
    """
    Compute the reconstruction deletion pattern as a list of
    (species_Z, frac_x, frac_y) tuples from the unit-cell data.

    This information can be reused to apply the same reconstruction
    to any plane with the same composition in a thicker slab.
    """
    frac = surf_bulk.get_scaled_positions()
    deleted_set = set(deletion_mask)
    delete_info = []
    for idx in cut_plane["indices"]:
        if idx in deleted_set:
            species = int(atoms_z_matrix[idx, 0])
            fx = float(frac[idx, 0]) % 1.0
            fy = float(frac[idx, 1]) % 1.0
            delete_info.append((species, fx, fy))
    return delete_info
