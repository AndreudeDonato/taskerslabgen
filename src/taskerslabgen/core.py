from collections import Counter
from functools import cmp_to_key
from math import gcd

import numpy as np
from ase.build import surface
from ase.data import atomic_numbers, chemical_symbols
from ase.formula import Formula
from ase.io import read
from scipy.optimize import linear_sum_assignment

# Single-linkage z-gap (angstrom) below which atoms share a plane; same
# default as pymatgen's ``ftol``.
DEFAULT_PLANE_TOL = 0.1

# Per-atom array used internally to track which input atom each slab atom
# came from (survives ``ase.build.surface`` and slicing).  Removed before
# slabs are returned.
_INDEX_KEY = "_tsg_index"


class SlabValidationError(ValueError):
    """A generated slab is not stoichiometric, neutral, non-polar and contiguous."""


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


def enumerate_cut_pairs(planes, L, reduced_counts, charge_tol=1e-3):
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
        Largest |net charge| per formula unit (e) treated as neutral.

    Returns
    -------
    list of dict
        Each entry describes a cut sequence with keys ``bottom_cut``,
        ``top_cut``, ``plane_indices``, ``total_charge``, ``net_dipole``,
        ``is_neutral``, ``is_stoich``, ``stoich_k``, ``dipole_per_fu``
        (|dipole| per formula unit, ``None`` if not stoichiometric),
        ``is_full_period`` (the sequence spans one whole bulk repeat unit),
        etc.
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
                    "is_neutral": abs(total_q) <= charge_tol * max(
                        1.0, sum(seq_counts.values()) / atoms_per_fu),
                    "is_stoich": is_stoich,
                    "stoich_k": stoich_k,
                    "dipole_per_fu": abs(mu_btt) / stoich_k if is_stoich else None,
                    "is_full_period": len(seq_indices_btt) == n,
                }
            )

    sequences.sort(key=lambda s: abs(s["net_dipole"]), reverse=True)
    return sequences


def select_best_sequence(sequences, dipole_tol=0.05):
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
        Largest |dipole| per formula unit (e·Å) of the repeat unit that is
        still considered zero (Tasker I/II).

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
        d = s["dipole_per_fu"]
        return (0.0 if d <= dipole_tol else d, s["bottom_cut"])

    best = dict(min(valid, key=key))
    best["is_tasker_ii"] = best["dipole_per_fu"] <= dipole_tol
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
                  dipole_tol=0.05, max_gap=None):
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
    charge_tol, dipole_tol : float
        Tolerances on the net charge (e) and on the dipole along *axis*
        (e·Å), both per formula unit of the slab.
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
    net_q = float(np.sum(q))
    if abs(net_q) > charge_tol * k:
        problems.append(f"net charge {net_q:+.4f} e")

    z = slab.positions[:, axis]
    dipole = float(np.sum(q * (z - z.mean())))
    if abs(dipole) > dipole_tol * k:
        problems.append(f"dipole {dipole:+.4e} e*A")

    if max_gap is not None and len(slab) > 1:
        gap = float(np.max(np.diff(np.sort(z))))
        if gap > max_gap + 1e-6:
            problems.append(f"internal z-gap of {gap:.2f} A (bulk max {max_gap:.2f} A)")

    if problems:
        raise SlabValidationError(
            f"Invalid slab {slab.get_chemical_formula()}: {'; '.join(problems)} "
            f"(charge_tol={charge_tol}, dipole_tol={dipole_tol} per formula unit)."
        )


def _finalize_slab(slab, charges_list, reduced_counts, charge_tol, dipole_tol,
                   max_gap=None, axis=2):
    """Validate a slab cut from an index-tagged structure, then drop the tag."""
    q = np.asarray(charges_list, dtype=float)[slab.arrays[_INDEX_KEY]]
    del slab.arrays[_INDEX_KEY]
    validate_slab(slab, q, reduced_counts, axis=axis, charge_tol=charge_tol,
                  dipole_tol=dipole_tol, max_gap=max_gap)


def assign_plane_names(planes_sorted, atoms=None, axis=2, xy_tol=0.5):
    """
    Label planes by their composition, e.g. ``O4``, ``Ce4``, ``Ir2O2``.

    A label depends only on the plane itself, so the same plane gets the
    same label in the bulk cell (genslab), in a slab cut from it (cutslab),
    and for any choice of bulk origin.  When one composition occurs in
    several geometries that are not related by an in-plane translation
    (e.g. the mirror-related IrO2 planes of rutile (001)), a variant letter
    is appended: ``IrO2-a``, ``IrO2-b``.  Variants are ordered by a
    translation-invariant fingerprint of their geometry
    (:func:`_plane_signature`), not by stacking order, so small
    displacements do not swap letters.

    Elements are written metals first, then non-metals, each alphabetically
    (ASE's ``"metal"`` formula format).  Reconstruction suffixes such as
    ``O4-recon`` are added by callers, not here.  Without *atoms*, planes are
    labelled by composition only.

    Parameters
    ----------
    planes_sorted : list of dict
        Planes from :func:`identify_planes`, in stacking order.
    atoms : Atoms or None
        Structure the plane indices refer to; enables geometric variants.
    axis : int
        Stacking axis.
    xy_tol : float
        Matching tolerance (angstrom, in-plane) for corresponding atoms,
        e.g. to absorb small relaxations.

    Returns
    -------
    names : list of str
        ``names[i]`` is the label of ``planes_sorted[i]``.
    name_map : dict
        ``{label: counts_dict}``.
    """
    import string

    if atoms is not None:
        ab_axes = [i for i in range(3) if i != axis]
        frac_all = atoms.get_scaled_positions()
        cell2d = np.array(atoms.cell)[np.ix_(ab_axes, ab_axes)]

    # Translation classes per composition: formula -> [[geometry, counts], ...]
    classes = {}
    class_of = []
    for plane in planes_sorted:
        formula = _formula_label(plane["counts"])
        groups = classes.setdefault(formula, [])
        if atoms is None:
            if not groups:
                groups.append([None, dict(plane["counts"])])
            class_of.append((formula, 0))
            continue
        geom = [
            (int(atoms.numbers[i]), frac_all[i, ab_axes[0]], frac_all[i, ab_axes[1]])
            for i in plane["indices"]
        ]
        for k, (ref_geom, _) in enumerate(groups):
            if _find_plane_translation(ref_geom, geom, cell2d, xy_tol) is not None:
                class_of.append((formula, k))
                break
        else:
            groups.append([geom, dict(plane["counts"])])
            class_of.append((formula, len(groups) - 1))

    labels = {}
    for formula, groups in classes.items():
        if len(groups) == 1:
            labels[(formula, 0)] = formula
            continue
        if len(groups) > len(string.ascii_lowercase):
            raise ValueError(f"Too many geometric variants of {formula} planes (more than 26).")
        sigs = [_plane_signature(groups[k][0]) for k in range(len(groups))]
        order = sorted(range(len(groups)),
                       key=cmp_to_key(lambda i, j: _compare_signatures(sigs[i], sigs[j])))
        for letter, k in zip(string.ascii_lowercase, order):
            labels[(formula, k)] = f"{formula}-{letter}"

    names = [labels[c] for c in class_of]
    name_map = {
        labels[(formula, k)]: counts
        for formula, groups in classes.items()
        for k, (_, counts) in enumerate(groups)
    }
    return names, name_map


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


def plane_name_base(name):
    """
    Return the composition part of a plane label.

    Examples: ``IrO2-a`` → ``IrO2``, ``O4-recon`` → ``O4``,
    ``IrO2-b-recon`` → ``IrO2``.
    """
    if not name:
        return name
    core = name[:-6] if name.endswith("-recon") else name
    core = core.rstrip("'")
    head, sep, tail = core.rpartition("-")
    if sep and len(tail) == 1 and tail.isalpha() and tail.islower():
        return head
    return core


def plane_name_matches(query, name):
    """
    Whether *query* selects plane label *name*.

    Matching rules:

    - exact equality (``O4-recon`` ↔ ``O4-recon``)
    - query equals name without ``-recon`` (``O4`` ↔ ``O4-recon``)
    - query equals name without the deformation prime (``O4`` ↔ ``O4'``)
    - query equals the composition (``IrO2`` ↔ ``IrO2-a``, ``IrO2-b``,
      ``IrO2-a-recon``, ``IrO2-b'``)
    """
    if query == name:
        return True
    if name.endswith("-recon") and query == name[:-6]:
        return True
    if query == name.rstrip("'"):
        return True
    return query == plane_name_base(name)


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
    if len(ref) != len(tgt):
        return []
    ref_Z = np.array([a[0] for a in ref], dtype=int)
    tgt_Z = np.array([a[0] for a in tgt], dtype=int)
    species, counts = np.unique(ref_Z, return_counts=True)
    tgt_species, tgt_counts = np.unique(tgt_Z, return_counts=True)
    if not (np.array_equal(species, tgt_species) and np.array_equal(counts, tgt_counts)):
        return []
    if len(ref) == 0:
        return [np.zeros(2)]

    ref_f = np.array([[a[1], a[2]] for a in ref], dtype=float)
    tgt_f = np.array([[a[1], a[2]] for a in tgt], dtype=float)
    cell2d = np.asarray(cell2d, dtype=float)
    anchor_Z = species[np.argmin(counts)]
    anchor = int(np.flatnonzero(ref_Z == anchor_Z)[0])
    found = []
    for b in np.flatnonzero(tgt_Z == anchor_Z):
        t = (tgt_f[b] - ref_f[anchor]) % 1.0
        if _shift_matches(ref, tgt, t, cell2d, tol) and not any(
            _frac_distance(t, u, cell2d) < 1e-3 for u in found
        ):
            found.append(t)
    return found


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


def _find_plane_translation(ref, tgt, cell2d, tol):
    """
    One in-plane translation mapping plane *ref* onto plane *tgt* (see
    :func:`_plane_translations`), or ``None``.
    """
    found = _plane_translations(ref, tgt, cell2d, tol)
    return found[0] if found else None


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
    labels, _ = assign_plane_names(planes, atoms=surf)
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
    ref_Z = np.array([a[0] for a in ref], dtype=int)
    tgt_Z = np.array([a[0] for a in tgt], dtype=int)
    species, counts = np.unique(ref_Z, return_counts=True)
    tgt_species, tgt_counts = np.unique(tgt_Z, return_counts=True)
    if len(ref) != len(tgt) or not (
        np.array_equal(species, tgt_species) and np.array_equal(counts, tgt_counts)
    ):
        return float("inf")
    ref_f = np.array([[a[1], a[2]] for a in ref], dtype=float)
    tgt_f = np.array([[a[1], a[2]] for a in tgt], dtype=float)
    ref_dz = np.array([a[3] for a in ref], dtype=float)
    tgt_dz = np.array([a[3] for a in tgt], dtype=float)
    ref_dz, tgt_dz = ref_dz - ref_dz.mean(), tgt_dz - tgt_dz.mean()
    cell2d = np.asarray(cell2d, dtype=float)
    groups = [(np.flatnonzero(ref_Z == Z), np.flatnonzero(tgt_Z == Z)) for Z in species]
    anchor = int(np.flatnonzero(ref_Z == species[np.argmin(counts)])[0])

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
    for b in np.flatnonzero(tgt_Z == species[np.argmin(counts)]):
        t = tgt_f[b] - ref_f[anchor]
        for _ in range(2):  # refine the shift by the mean residual
            res = residuals(t)
            t = t + np.mean([d for d, _ in res], axis=0)
        res = residuals(t)
        msd = np.mean([np.sum((d @ cell2d) ** 2) + dz ** 2 for d, dz in res])
        best = min(best, float(np.sqrt(msd)))
    return best


def _catalog_in_cell(catalog, bulk_cell2d, cell2d, L, site_tol=0.3):
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
        planes.append({"indices": list(range(start, len(numbers))), "counts": entry["counts"]})
        start = len(numbers)
    cell = np.zeros((3, 3))
    cell[:2, :2] = cell2d
    cell[2, 2] = L
    folded = Atoms(numbers=numbers, scaled_positions=scaled, cell=cell, pbc=True)
    labels, _ = assign_plane_names(planes, atoms=folded)
    for entry, label in zip(catalog, labels):
        entry["label"] = label


def _planes_from_bulk(atoms, charges_list, bulk_atoms, miller, plane_tol=None,
                      charge_tol=1e-3, deform_tol=0.3):
    """
    Assign the atoms of a slab to the planes of its bulk and label them.

    The vertical registry between slab and bulk is learned from the slab's
    bulk-like interior; every atom then goes to the nearest bulk plane that
    contains its species, so rumpled or relaxed surface planes stay whole.
    A plane gets the bulk label (e.g. ``O4``) when it matches its bulk plane
    within *deform_tol* (RMSD in angstrom after the best rigid shift) and a
    primed label (``O4'``) when it is more deformed or has a different
    composition.

    Returns ``(planes_sorted, labels, reduced_counts)`` with planes in the
    format of :func:`identify_planes`.
    """
    catalog, L, bulk_cell2d = _bulk_plane_catalog(bulk_atoms, miller, plane_tol)
    cell2d = np.array(atoms.cell[:2, :2])
    _catalog_in_cell(catalog, bulk_cell2d, cell2d, L)

    numbers = atoms.numbers
    z = atoms.positions[:, 2]
    q = np.asarray(charges_list, dtype=float)
    atoms_z = np.column_stack([numbers, z, q])
    slab_planes = sorted(
        identify_planes(atoms_z, float(atoms.cell[2, 2]), plane_tol=plane_tol),
        key=lambda p: p["z_center"],
    )

    def wrap(dz):
        return ((dz + 0.5 * L) % L) - 0.5 * L

    # Registry offset: z_slab = z_bulk + offset (mod L), scored on all planes,
    # candidates taken from the bulk-like middle of the slab.
    n_sp = len(slab_planes)
    middle = slab_planes[n_sp // 4: n_sp - n_sp // 4] or slab_planes
    candidates = [
        (p["z_center"] - b["z"]) % L
        for p in middle for b in catalog if b["counts"] == p["counts"]
    ]
    # Atoms of the middle half of the slab, for registering single atoms when
    # relaxation split every plane (no slab plane has a bulk composition).
    span = z.max() - z.min()
    inner = np.flatnonzero(np.abs(z - (z.min() + 0.5 * span)) <= 0.25 * span + 1e-9)
    species_planes = {
        int(Zi): np.array([b["z"] for b in catalog if int(Zi) in b["counts"]])
        for Zi in set(numbers.tolist())
    }
    if any(len(v) == 0 for v in species_planes.values()):
        raise ValueError(
            f"The slab contains elements absent from the {tuple(miller)} planes of the "
            "bulk; check bulk_atoms and miller."
        )

    def atom_cost(offset):
        """Summed distance of the inner atoms to the nearest bulk plane of their species."""
        return float(sum(
            np.min(np.abs(((z[i] - species_planes[int(numbers[i])] - offset + 0.5 * L) % L) - 0.5 * L))
            for i in inner
        ))

    frac = atoms.get_scaled_positions()

    def geometry(idx, zc):
        return [(int(numbers[i]), frac[i, 0], frac[i, 1], z[i] - zc) for i in idx]

    slab_geoms = [geometry(p["indices"], p["z_center"]) for p in slab_planes]

    def matched(offset):
        """(slab plane, bulk plane, rmsd) pairs aligned by this offset."""
        return [
            (k, b, _plane_rmsd(b["atoms"], slab_geoms[k], cell2d))
            for k, p in enumerate(slab_planes) for b in catalog
            if b["counts"] == p["counts"] and abs(wrap(p["z_center"] - b["z"] - offset)) < 0.3
        ]

    def score(offset):
        # Atoms in planes that match their bulk plane geometrically; planes of
        # equal composition (e.g. mirror variants) are told apart this way.
        good = [(len(slab_planes[k]["indices"]), r) for k, _, r in matched(offset) if r <= deform_tol]
        return (sum(n for n, _ in good), -sum(r for _, r in good))

    if candidates:
        candidates = sorted({round(c, 4) for c in candidates})
        offset = max(candidates, key=score)
        in_middle = {id(p) for p in middle}
        pairs = [
            (slab_planes[k], b) for k, b, r in matched(offset)
            if r <= deform_tol and id(slab_planes[k]) in in_middle
        ] or [(slab_planes[k], b) for k, b, _ in matched(offset)]
        z_ref = [b["z"] for _, b in pairs]
        z_slab = np.array([p["z_center"] for p, _ in pairs])
    else:
        candidates = sorted({
            round((z[i] - zb) % L, 4) for i in inner for zb in species_planes[int(numbers[i])]
        })
        offset = min(candidates, key=atom_cost)
        z_ref = []
        for i in inner:
            zb = species_planes[int(numbers[i])]
            d = ((z[i] - zb - offset + 0.5 * L) % L) - 0.5 * L
            z_ref.append(float(zb[np.argmin(np.abs(d))]))
        z_slab = z[inner]
    # z_slab = offset + scale * z_bulk, fitted on the bulk-like interior: the
    # slab may be strained along the normal relative to bulk_atoms (e.g.
    # built from another calculation), which adds up over many layers.
    z_ref = np.asarray(z_ref, dtype=float)
    z_bulk = z_ref + L * np.round((z_slab - z_ref - offset) / L)
    if np.ptp(z_bulk) > 0.5 * L:
        scale, offset = (float(v) for v in np.polyfit(z_bulk, z_slab, 1))
    else:
        scale, offset = 1.0, float(np.mean(z_slab - z_bulk))

    # Species-aware nearest bulk plane for every atom.
    groups = {}
    for i, (Zi, zi) in enumerate(zip(numbers, z)):
        options = [k for k, b in enumerate(catalog) if int(Zi) in b["counts"]] or range(len(catalog))
        best = None
        for k in options:
            m = int(np.round(((zi - offset) / scale - catalog[k]["z"]) / L))
            dist = abs(zi - (offset + scale * (catalog[k]["z"] + m * L)))
            if best is None or dist < best[0]:
                best = (dist, k, m)
        groups.setdefault((best[1], best[2]), []).append(i)

    keyed = sorted(groups.items(), key=lambda kv: catalog[kv[0][0]]["z"] + kv[0][1] * L)
    planes_sorted, labels = [], []
    for (k, _), idx in keyed:
        plane = _make_plane(idx, z[idx], atoms_z, charge_tol)
        rmsd = _plane_rmsd(catalog[k]["atoms"], geometry(idx, plane["z_center"]), cell2d)
        planes_sorted.append(plane)
        labels.append(catalog[k]["label"] + ("" if rmsd <= deform_tol else "'"))

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


def extract_termination(reference, charges, axis=2, plane_tol=None, charge_tol=1e-3):
    """
    Extract termination fingerprints from a reference slab.

    Parameters
    ----------
    reference : Atoms, str/Path, or dict (genslab termination entry)
        - Atoms / file: extract bottom/top fingerprints only.
        - Dict with ``"tasker_type"`` key: also extract reconstruction
          metadata so cutslab can reapply the same Tasker III pattern.

    Returns
    -------
    dict with keys: ``"bottom"``, ``"top"``, ``"reconstruction"``,
    ``"plane_names"``, ``"plane_name_map"``.
    """
    reconstruction = None
    plane_names = None
    plane_name_map = None

    if isinstance(reference, dict) and "tasker_type" in reference:
        if "atoms" in reference:
            slab = reference["atoms"][0]
        elif "slab_atoms" in reference:
            slab = reference["slab_atoms"][0]
        else:
            raise ValueError("Termination dict must contain 'atoms' or 'slab_atoms'")

        recon = reference.get("reconstruction")
        if recon is not None:
            reconstruction = recon
            plane_names = recon.get("plane_names")
            plane_name_map = recon.get("plane_name_map")

        if plane_names is None and "plane_classification" in reference:
            plane_names = reference["plane_classification"].get("plane_names")
            plane_name_map = reference["plane_classification"].get("plane_name_map")

        reference = slab

    if not hasattr(reference, "positions"):
        reference = read(str(reference))

    charges_list = _charges_to_list(reference, charges)

    L = float(reference.cell.lengths()[axis])
    if L <= 0.0:
        raise ValueError("Invalid cell length on selected axis.")

    coords = reference.positions[:, axis]
    atoms_z = np.array(
        [[num, z, q] for num, z, q in zip(reference.numbers, coords, charges_list)]
    )

    planes = identify_planes(atoms_z, L, plane_tol=plane_tol, charge_tol=charge_tol)
    planes_sorted = sorted(planes, key=lambda p: p["z_center"] % L)

    frac_all = reference.get_scaled_positions()

    ab_axes = [i for i in range(3) if i != axis]

    def _fingerprint(plane):
        indices = plane["indices"]
        frac_xy = []
        for idx in indices:
            Z = int(atoms_z[idx, 0])
            fx = float(frac_all[idx, ab_axes[0]]) % 1.0
            fy = float(frac_all[idx, ab_axes[1]]) % 1.0
            frac_xy.append((Z, fx, fy))
        return {"counts": dict(plane["counts"]), "frac_xy": frac_xy}

    bottom_plane = planes_sorted[0]
    top_plane = planes_sorted[-1]

    return {
        "bottom": _fingerprint(bottom_plane),
        "top": _fingerprint(top_plane),
        "reconstruction": reconstruction,
        "plane_names": plane_names,
        "plane_name_map": plane_name_map,
    }


def plane_match_score(plane, ref_fingerprint, atoms, axis=2):
    """
    Score how well a candidate plane matches a reference termination
    fingerprint.

    Hard constraint: elemental composition must be identical.
    Soft score: RMSD of fractional xy positions using Hungarian
    assignment with PBC wrapping.

    Returns (matches, rmsd).
    """
    if plane["counts"] != ref_fingerprint["counts"]:
        return False, float("inf")

    ab_axes = [i for i in range(3) if i != axis]
    frac_all = atoms.get_scaled_positions()

    ref_by_species = {}
    for Z, fx, fy in ref_fingerprint["frac_xy"]:
        ref_by_species.setdefault(Z, []).append((fx, fy))

    cand_by_species = {}
    for idx in plane["indices"]:
        Z = int(atoms.numbers[idx])
        fx = float(frac_all[idx, ab_axes[0]]) % 1.0
        fy = float(frac_all[idx, ab_axes[1]]) % 1.0
        cand_by_species.setdefault(Z, []).append((fx, fy))

    total_sq = 0.0
    total_count = 0

    for Z, ref_pts in ref_by_species.items():
        cand_pts = cand_by_species.get(Z, [])
        if len(ref_pts) != len(cand_pts):
            return False, float("inf")
        n = len(ref_pts)
        cost = np.zeros((n, n))
        for i, (rx, ry) in enumerate(ref_pts):
            for j, (cx, cy) in enumerate(cand_pts):
                dx = abs(rx - cx)
                dy = abs(ry - cy)
                dx = min(dx, 1.0 - dx)
                dy = min(dy, 1.0 - dy)
                cost[i, j] = dx * dx + dy * dy
        row_ind, col_ind = linear_sum_assignment(cost)
        total_sq += float(cost[row_ind, col_ind].sum())
        total_count += n

    if total_count == 0:
        return True, 0.0

    rmsd = np.sqrt(total_sq / total_count)
    return True, rmsd
