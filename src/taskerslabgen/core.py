from itertools import product
from math import gcd

import numpy as np
from ase.build import surface
from ase.data import atomic_numbers
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
        returns the bare oriented bulk: atoms are not shifted and the third
        cell vector is normal to the surface with length ``layers * L``.
    verbose : bool or None
        Print debug information.

    Returns
    -------
    Atoms
        Surface slab with full PBC enabled.  Its third cell vector is not a
        bulk lattice vector in general; see :func:`surface_bulk_cell`.
    """
    _check_miller(miller)
    if vacuum is None or vacuum <= 0:
        # Not ``periodic=True``: ASE would then wrap atoms along the normal by
        # its orthogonalised third vector, which is not a lattice vector.
        slab = surface(bulk_atoms, miller, layers=layers)
        area = np.linalg.norm(np.cross(slab.cell[0], slab.cell[1]))
        slab.cell[2] = [0.0, 0.0, layers * bulk_atoms.get_volume() / area]
    else:
        slab = surface(bulk_atoms, miller, layers=layers, vacuum=vacuum)
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
    charges : dict or list
        Formal charges.  A dict maps element symbols or atomic numbers to
        charge values; a list gives per-atom charges.
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
    if isinstance(charges, dict):
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


def _charges_to_list(atoms, charges):
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


def _make_plane(indices, z_center, atoms_z, charge_tol):
    q_total = float(np.sum(atoms_z[indices, 2]))
    if abs(q_total) < charge_tol:
        q_total = 0.0
    counts = {}
    for Z in atoms_z[indices, 0].astype(int):
        counts[int(Z)] = counts.get(int(Z), 0) + 1
    return {
        "z_center": float(z_center),
        "q_total": q_total,
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
        Charges with ``abs(q) < charge_tol`` are set to exactly 0.

    Returns
    -------
    list of dict
        Planes sorted by ``z_center`` (in ``[0, L)``).  Each dict contains
        ``z_center``, ``q_total``, ``indices``, and ``counts`` (element
        composition ``{Z: count}``).
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
            planes.append(_make_plane(group, np.mean(group_z) % L, atoms_z, charge_tol))
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
        Tolerance for charge neutrality.

    Returns
    -------
    list of dict
        Each entry describes a cut sequence with keys ``bottom_cut``,
        ``top_cut``, ``plane_indices``, ``total_charge``, ``net_dipole``,
        ``is_neutral``, ``is_stoich``, ``stoich_k``, ``is_full_period``
        (the sequence spans one whole bulk repeat unit), etc.
    """
    if len(planes) == 0:
        return []

    planes_sorted = sorted(planes, key=lambda p: p["z_center"] % L)
    z_sorted = np.array([p["z_center"] % L for p in planes_sorted], dtype=float)
    q_sorted = np.array([p["q_total"] for p in planes_sorted], dtype=float)
    counts_sorted = [p["counts"] for p in planes_sorted]
    n = len(planes_sorted)

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
            mu_btt = float(np.sum(q_seq_btt * (z_seq_btt - z_center_btt)))
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
                    "is_neutral": abs(total_q) <= charge_tol,
                    "is_stoich": is_stoich,
                    "stoich_k": stoich_k,
                    "is_full_period": len(seq_indices_btt) == n,
                }
            )

    sequences.sort(key=lambda s: abs(s["net_dipole"]), reverse=True)
    return sequences


def select_best_sequence(sequences, dipole_tol=1e-6):
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
        Threshold below which the dipole is considered zero (Tasker I/II).

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
        mu = abs(s["net_dipole"])
        return (0.0 if mu <= dipole_tol else mu, s["bottom_cut"])

    best = dict(min(valid, key=key))
    best["is_tasker_ii"] = abs(best["net_dipole"]) <= dipole_tol
    return best


def compute_cut_positions(planes, L, bottom_cut_index, top_cut_index):
    """
    Compute z-coordinates for the bottom and top cuts (midpoints between
    adjacent planes).

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
    z_sorted = np.array([p["z_center"] % L for p in planes_sorted], dtype=float)
    n = len(z_sorted)

    def midpoint(i):
        z0 = z_sorted[i]
        z1 = z_sorted[(i + 1) % n]
        if z1 <= z0:  # wraps (or a single plane: next copy is one L above)
            z1 += L
        return 0.5 * (z0 + z1)

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
    atoms.set_positions(positions)
    cell = atoms.get_cell().copy()
    vec = np.zeros(3)
    vec[axis] = new_height
    cell[axis] = vec
    atoms.set_cell(cell)
    atoms.set_pbc([True, True, True])


def validate_slab(slab, charges, reduced_counts, axis=2, charge_tol=1e-3,
                  dipole_tol=1e-6, max_gap=None):
    """
    Check that *slab* satisfies the conditions the generators promise.

    Parameters
    ----------
    slab : Atoms
        Slab to check.
    charges : dict or list
        Charges by element, or one charge per atom of *slab*.
    reduced_counts : dict
        Reduced bulk stoichiometry ``{Z: count}``.
    axis : int
        Surface-normal axis.
    charge_tol, dipole_tol : float
        Tolerances on the net charge (e) and on the dipole along *axis*
        (e·Å), both per formula unit of the slab.
    max_gap : float or None
        Largest allowed z-gap (angstrom) between neighbouring atoms; catches
        slabs glued together across vacuum.  ``None`` skips the check.

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


def _max_z_gap(z, period=None):
    """Largest gap between sorted z values (wrapping through *period* if given)."""
    z = np.asarray(z, dtype=float)
    if len(z) < 2:
        return None
    if period is None:
        z = np.sort(z)
    else:
        z = np.sort(z % period)
        z = np.append(z, z[0] + period)
    return float(np.max(np.diff(z)))


def assign_plane_names(planes_sorted, atoms=None, axis=2, xy_tol=0.5):
    """
    Assign hierarchical plane labels ``P{n}{letter}`` (e.g. ``P0a``, ``P0b``).

    - **Type** ``P{n}``: same composition, and in-plane geometry that maps
      onto each other under a point-group operation of the in-plane lattice
      plus a translation (8 operations for a square lattice, 12 hexagonal,
      4 rectangular, 2 oblique).
    - **Variant letter**: planes related by a pure in-plane translation
      share the full name; congruent planes that are not translation-related
      get the next letter (``a``, ``b``, …) in order of first appearance
      along the stacking axis.

    Reconstruction suffixes such as ``P0a-recon`` are added by callers, not
    here.  Without *atoms*, planes are named by composition only.

    Parameters
    ----------
    planes_sorted : list of dict
        Planes from :func:`identify_planes`, in stacking order.
    atoms : Atoms or None
        Structure the plane indices refer to; enables geometric matching.
    axis : int
        Stacking axis.
    xy_tol : float
        Matching tolerance (angstrom, in-plane) for corresponding atoms,
        e.g. to absorb small relaxations.

    Returns
    -------
    names : list of str
        ``names[i]`` is the name of ``planes_sorted[i]``.
    name_map : dict
        ``{name: counts_dict}``.
    """
    import string

    identity = [np.eye(2, dtype=int)]
    lattice_ops = identity
    if atoms is not None:
        ab_axes = [i for i in range(3) if i != axis]
        frac_all = atoms.get_scaled_positions()
        cell2d = np.array(atoms.cell)[np.ix_(ab_axes, ab_axes)]
        lattice_ops = _lattice_point_ops(cell2d)

    def geometry(plane):
        if atoms is None:
            return None
        return [
            (int(atoms.numbers[i]), frac_all[i, ab_axes[0]], frac_all[i, ab_axes[1]])
            for i in plane["indices"]
        ]

    def related(geom_a, geom_b, ops):
        if geom_a is None:
            return True
        return _find_plane_alignment(geom_a, geom_b, cell2d, xy_tol, ops) is not None

    # Per type index: list of (counts, geometry, full_name)
    type_variants = []
    next_letter = []
    names = []
    name_map = {}

    for plane in planes_sorted:
        counts = dict(plane["counts"])
        geom = geometry(plane)

        matched_name = None
        matched_type = None
        for t_idx, variants in enumerate(type_variants):
            if variants[0][0] != counts:
                continue
            for _, vgeom, vname in variants:
                if related(vgeom, geom, identity):
                    matched_name = vname
                    break
            if matched_name is not None:
                break
            if any(related(vgeom, geom, lattice_ops) for _, vgeom, _ in variants):
                matched_type = t_idx
                break

        if matched_name is not None:
            names.append(matched_name)
            continue

        if matched_type is not None:
            letter_i = next_letter[matched_type]
            if letter_i >= len(string.ascii_lowercase):
                raise ValueError(
                    f"Too many plane variants for type P{matched_type} "
                    f"(exceeded 26 letters)."
                )
            name = f"P{matched_type}{string.ascii_lowercase[letter_i]}"
            next_letter[matched_type] = letter_i + 1
            type_variants[matched_type].append((counts, geom, name))
            name_map[name] = counts
            names.append(name)
            continue

        # New type
        t_idx = len(type_variants)
        name = f"P{t_idx}a"
        type_variants.append([(counts, geom, name)])
        next_letter.append(1)
        name_map[name] = counts
        names.append(name)

    return names, name_map


def plane_name_base(name):
    """
    Return the type base of a plane label.

    Examples: ``P0a`` → ``P0``, ``P0a-recon`` → ``P0``, ``P12b`` → ``P12``.
    """
    if not name:
        return name
    core = name[:-6] if name.endswith("-recon") else name
    if len(core) >= 2 and core[0] == "P":
        # Strip trailing variant letter if present (P0a, P12b, …)
        if core[-1].isalpha() and core[-1].islower():
            digits = core[1:-1]
            if digits.isdigit():
                return f"P{digits}"
        # Bare type like P0 (no letter)
        if core[1:].isdigit():
            return core
    return core


def plane_name_matches(query, name):
    """
    Whether *query* selects plane label *name*.

    Matching rules:

    - exact equality (``P0a-recon`` ↔ ``P0a-recon``)
    - query equals name without ``-recon`` (``P0a`` ↔ ``P0a-recon``)
    - query equals the type base (``P0`` ↔ ``P0a``, ``P0b``, ``P0a-recon``)
    """
    if query == name:
        return True
    if name.endswith("-recon") and query == name[:-6]:
        return True
    return query == plane_name_base(name)


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
    Point-group operations of the 2D lattice with basis rows *cell2d*.

    Returned as integer matrices ``W`` acting on fractional coordinates
    (``f' = f @ W``): 8 for a square lattice, 12 hexagonal, 4 (centred)
    rectangular, 2 oblique.
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


def _find_plane_alignment(ref, tgt, cell2d, tol, ops=None):
    """
    Find an in-plane operation mapping plane *ref* onto plane *tgt*.

    *ref* and *tgt* are lists of ``(Z, fx, fy)`` in fractional coordinates
    of the in-plane lattice with basis rows *cell2d*.  Returns ``(W, t)``
    such that every ref atom moved to ``f @ W + t`` lies within *tol*
    angstrom of a distinct tgt atom of the same species, or ``None``.
    *ops* defaults to the identity (pure translations).
    """
    if len(ref) != len(tgt):
        return None
    ref_Z = np.array([a[0] for a in ref], dtype=int)
    tgt_Z = np.array([a[0] for a in tgt], dtype=int)
    species, counts = np.unique(ref_Z, return_counts=True)
    tgt_species, tgt_counts = np.unique(tgt_Z, return_counts=True)
    if not (np.array_equal(species, tgt_species) and np.array_equal(counts, tgt_counts)):
        return None
    if len(ref) == 0:
        return np.eye(2, dtype=int), np.zeros(2)

    ref_f = np.array([[a[1], a[2]] for a in ref], dtype=float)
    tgt_f = np.array([[a[1], a[2]] for a in tgt], dtype=float)
    cell2d = np.asarray(cell2d, dtype=float)
    anchor_Z = species[np.argmin(counts)]
    anchor = int(np.flatnonzero(ref_Z == anchor_Z)[0])
    groups = [(np.flatnonzero(ref_Z == Z), np.flatnonzero(tgt_Z == Z)) for Z in species]

    for W in (ops if ops is not None else [np.eye(2, dtype=int)]):
        moved = ref_f @ W
        for b in np.flatnonzero(tgt_Z == anchor_Z):
            t = tgt_f[b] - moved[anchor]
            for r_idx, t_idx in groups:
                d = (moved[r_idx] + t)[:, None, :] - tgt_f[t_idx][None, :, :]
                d -= np.round(d)
                dist = np.linalg.norm(d @ cell2d, axis=-1)
                rows, cols = linear_sum_assignment(dist)
                if dist[rows, cols].max() > tol:
                    break
            else:
                return W, t % 1.0
    return None


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
