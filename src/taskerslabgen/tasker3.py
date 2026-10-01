import numpy as np
from itertools import combinations, product
from math import comb

from ase import Atoms
from ase.data import atomic_numbers, covalent_radii, chemical_symbols
from ase.neighborlist import neighbor_list

from .core import (
    _INDEX_KEY,
    _charges_to_list,
    _finalize_slab,
    _formula_label,
    _oriented_bulk,
    _tag_atom_indices,
    build_surface,
    compute_projection,
    identify_planes,
    compute_reduced_counts,
    assign_plane_names,
    apply_vacuum_to_slab,
    compute_cut_positions,
    compute_delete_info,
    plane_name_matches,
    surface_bulk_cell,
)


def build_adjacency_matrix(atoms, bond_threshold=(0.85, 1.15), bond_distances=None,
                           bulk_atoms=None, miller=None):
    """
    Build a bond-count matrix using covalent radii and PBC.

    ``adj[i, j]`` is the number of periodic images of atom *j* bonded to
    atom *i*: their distance falls within ``[lo * d_ref, hi * d_ref]`` where
    ``d_ref`` is either the user-supplied reference or the sum of covalent
    radii.  Counting images keeps bond counts right in small cells; use
    ``adj > 0`` for a boolean view.

    Parameters
    ----------
    atoms : Atoms
        The structure whose atom indices will label the matrix.
    bond_threshold : tuple of float
        ``(lo, hi)`` scaling factors applied to the reference distance.
    bond_distances : dict, optional
        Per-pair reference distances.
        Keys: ``"Ce-O"`` style strings (order irrelevant) or tuples of
        atomic numbers.
        Values: ``float`` (scaled by *bond_threshold*) or ``None``
        (forbid that pair entirely).
    bulk_atoms : Atoms, optional
        Original bulk cell, for *atoms* built by
        ``build_surface(bulk_atoms, miller)``.  Periodic images then follow
        the true bulk lattice (:func:`surface_bulk_cell`) instead of the
        surface cell, whose third vector is not a bulk lattice vector.
    miller : tuple of int, optional
        Miller index of *atoms*; required with *bulk_atoms*.

    Returns
    -------
    ndarray of int, shape (N, N)
        Symmetric bond-count matrix.
    """
    if bulk_atoms is not None:
        if miller is None:
            raise ValueError(
                "bulk_atoms requires miller: the bulk lattice must be rotated into "
                "the surface frame (see surface_bulk_cell)."
            )
        atoms = Atoms(
            numbers=atoms.numbers,
            positions=atoms.get_positions(),
            cell=surface_bulk_cell(bulk_atoms, miller),
            pbc=True,
        )

    n = len(atoms)
    adj = np.zeros((n, n), dtype=int)
    i_idx, j_idx, _ = _bond_pairs(atoms, bond_threshold, bond_distances)
    np.add.at(adj, (i_idx, j_idx), 1)
    return adj


def _bond_pairs(atoms, bond_threshold=(0.85, 1.15), bond_distances=None):
    """
    Bonded pairs over periodic images, with the rules of
    :func:`build_adjacency_matrix`.

    Returns ``(i, j, D)``: every bond appears as ``(i, j)`` and ``(j, i)``;
    ``D`` is the vector from atom *i* to the bonded image of atom *j*.
    """
    lo, hi = bond_threshold
    manual_map = _parse_bond_distances_map(bond_distances)

    species = sorted({int(z) for z in atoms.numbers})
    slot = {z: k for k, z in enumerate(species)}
    ref = np.full((len(species), len(species)), np.nan)
    for za in species:
        for zb in species:
            pair = (min(za, zb), max(za, zb))
            d = manual_map.get(pair, covalent_radii[za] + covalent_radii[zb])
            if d is not None:
                ref[slot[za], slot[zb]] = d
    if np.all(np.isnan(ref)):
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros((0, 3))

    i_idx, j_idx, vec = neighbor_list("ijD", atoms, hi * float(np.nanmax(ref)))
    dist = np.linalg.norm(vec, axis=1)
    kinds = np.array([slot[int(z)] for z in atoms.numbers])
    pair_ref = ref[kinds[i_idx], kinds[j_idx]]
    with np.errstate(invalid="ignore"):
        bonded = (dist >= lo * pair_ref) & (dist <= hi * pair_ref)
    return i_idx[bonded], j_idx[bonded], vec[bonded]


def _bonds_across_plane(atoms, z_cut, period, bond_threshold=(0.85, 1.15),
                        bond_distances=None):
    """
    Bonds per cell crossing the planes ``z = z_cut + m * period``, by pair.

    *atoms* must be periodic with its true lattice (e.g. the 1-layer cell
    with :func:`surface_bulk_cell`); *period* is the layer spacing L.
    Returns ``{"Ce-O": 8, "Ce-Ce": 12, ...}``; the sum is the number of
    bonds a cut at *z_cut* breaks per surface cell.
    """
    i_idx, j_idx, vec = _bond_pairs(atoms, bond_threshold, bond_distances)
    z0 = atoms.positions[i_idx, 2]
    z1 = z0 + vec[:, 2]
    lower, upper = np.minimum(z0, z1), np.maximum(z0, z1)
    crossings = np.floor((upper - z_cut) / period) - np.floor((lower - z_cut) / period)
    by_pair = {}
    for a, b, c in zip(i_idx, j_idx, crossings):
        if c:
            key = "-".join(sorted((chemical_symbols[atoms.numbers[a]],
                                   chemical_symbols[atoms.numbers[b]])))
            by_pair[key] = by_pair.get(key, 0.0) + c
    return {k: int(round(v / 2.0)) for k, v in sorted(by_pair.items())}  # each bond listed twice


def print_adjacency_matrix(adj, atoms):
    """
    Print the adjacency matrix with element labels as row/column headers.

    Parameters
    ----------
    adj : ndarray
        Boolean adjacency matrix.
    atoms : Atoms
        Corresponding atomic structure (used for labels).
    """
    n = len(atoms)
    labels = []
    elem_count = {}
    for i in range(n):
        sym = chemical_symbols[atoms.numbers[i]]
        idx = elem_count.get(sym, 0)
        elem_count[sym] = idx + 1
        labels.append(f"{sym}{idx}")

    col_w = max(len(lb) for lb in labels) + 1
    header = " " * (col_w + 1) + "".join(lb.rjust(col_w) for lb in labels)
    print("\nAdjacency matrix:")
    print(header)
    for i in range(n):
        row = labels[i].rjust(col_w) + " "
        row += "".join(str(int(adj[i, j])).rjust(col_w) for j in range(n))
        print(row)
    print()


def _compute_plane_excess(plane_counts, reduced_counts):
    """
    Compute atoms to delete *per side* so the extra surface plane
    doesn't break stoichiometry.

    A slab with surface plane P has: N * bulk + P_comp (extra).
    We remove excess_per_side from *each* surface (symmetric),
    so total deletion = 2 * excess_per_side.

    We need:  P_comp - 2 * excess_per_side = j * reduced
    with j as large as possible, excess non-negative and integer.

    Returns (excess_per_side_dict, j) or (None, None) if impossible.
    """
    all_elements = set(list(plane_counts.keys()) + list(reduced_counts.keys()))

    j_max = float("inf")
    for Z, r in reduced_counts.items():
        if r == 0:
            continue
        p = plane_counts.get(Z, 0)
        j_max = min(j_max, p // r)
    if j_max == float("inf"):
        j_max = 0

    for j in range(int(j_max), -1, -1):
        excess = {}
        valid = True
        for Z in all_elements:
            p = plane_counts.get(Z, 0)
            r = reduced_counts.get(Z, 0)
            e = p - j * r
            if e < 0 or e % 2 != 0:
                valid = False
                break
            excess[Z] = e // 2
        if valid:
            return excess, j

    return None, None


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


def _stacking_symmetry(numbers, positions, cell, tol=0.1):
    """
    Atom permutations of the symmetry operations of a crystal that keep the
    stacking direction (in-plane rotation or mirror, any translation, z only
    shifted).

    *positions* and the true lattice *cell* (rows ``a1``, ``a2`` in-plane,
    ``a3``) describe one bulk repeat unit, e.g. the one-layer cell of
    :func:`build_surface` with :func:`surface_bulk_cell`.  Returns a list of
    arrays ``perm`` (``perm[a]`` is the image of atom ``a``), identity first.
    """
    cell = np.asarray(cell, dtype=float)
    numbers = np.asarray(numbers)
    frac = np.asarray(positions, dtype=float) @ np.linalg.inv(cell)
    cell2d = cell[:2, :2]
    inv2d = np.linalg.inv(cell2d)
    species, counts = np.unique(numbers, return_counts=True)
    anchor = int(np.flatnonzero(numbers == species[np.argmin(counts)])[0])
    perms = [np.arange(len(numbers))]
    seen = {tuple(perms[0])}
    for W in _lattice_point_ops(cell2d):
        # In-plane rotation R (Cartesian): cell2d @ R.T = W @ cell2d.  It is a
        # lattice operation only if it moves a3 by an in-plane lattice vector.
        RT = inv2d @ W @ cell2d
        k = (cell[2, :2] @ RT - cell[2, :2]) @ inv2d
        if not np.allclose(k, np.round(k), atol=1e-3):
            continue
        W3 = np.eye(3)
        W3[:2, :2] = W
        W3[2, :2] = np.round(k)
        rotated = frac @ W3
        for b in np.flatnonzero(numbers == numbers[anchor]):
            t = frac[b] - rotated[anchor]
            d = (rotated + t)[:, None, :] - frac[None, :, :]
            d -= np.round(d)
            dist = np.linalg.norm(d @ cell, axis=-1)
            dist[numbers[:, None] != numbers[None, :]] = np.inf
            perm = np.argmin(dist, axis=1)
            if dist[np.arange(len(perm)), perm].max() > tol or len(set(perm)) != len(perm):
                continue
            if tuple(perm) not in seen:
                seen.add(tuple(perm))
                perms.append(perm)
    return perms


def _enumerate_deletion_masks(plane_indices, atoms_z_matrix, excess):
    """
    Enumerate all ways to delete 'excess' atoms from a plane.
    Returns list of tuples of atom indices to delete.
    """
    if all(v == 0 for v in excess.values()):
        return [()]

    groups = {}
    for idx in plane_indices:
        Z = int(atoms_z_matrix[idx, 0])
        groups.setdefault(Z, []).append(idx)

    per_element = []
    for Z, n_del in excess.items():
        if n_del == 0:
            continue
        available = groups.get(Z, [])
        if len(available) < n_del:
            return []
        per_element.append(list(combinations(available, n_del)))

    if not per_element:
        return [()]

    results = [()]
    for combos in per_element:
        new_results = []
        for existing in results:
            for combo in combos:
                new_results.append(existing + combo)
        results = new_results

    return results


def _parse_bond_distances_map(bond_distances):
    """Convert user-facing bond_distances dict to a {(Zmin,Zmax): value} map."""
    from ase.data import atomic_numbers as ase_atomic_numbers

    manual_map = {}
    if bond_distances:
        for key, d in bond_distances.items():
            if isinstance(key, str):
                parts = key.split("-")
                if len(parts) != 2:
                    raise ValueError(f"Bond key must be 'X-Y', got: {key!r}")
                a = ase_atomic_numbers[parts[0].strip()]
                b = ase_atomic_numbers[parts[1].strip()]
            else:
                a, b = key
                if isinstance(a, str):
                    a = ase_atomic_numbers[a]
                if isinstance(b, str):
                    b = ase_atomic_numbers[b]
            manual_map[(min(a, b), max(a, b))] = d
    return manual_map


def _dangling_terms(r, i_idx, j_idx, dz, L):
    """
    Mask-independent terms for the bonds broken at one surface.

    *r* are the heights of the cell's atoms above the cut, in ``[0, L)``;
    ``(i_idx, j_idx, dz)`` are the bonds over periodic images (both
    directions), *dz* the height of the bonded image of *j* above *i*.  The
    slab holds the copies above the cut.  Each bond's partner sits ``p``
    periods above the copy of *j* in period 0.
    """
    p = np.rint((r[i_idx] + dz - r[j_idx]) / L).astype(int)
    n = len(r)
    cut = int(np.sum(np.maximum(0, -p)))                # bonds crossing the cut
    cross0 = np.bincount(i_idx[p <= -1], minlength=n)   # period-0 atom bonded below the cut
    inward = np.bincount(j_idx[p <= 0], minlength=n)    # bonds reaching the period-0 atom from the slab side
    pair0 = {}
    for a, b in zip(i_idx[p == 0], j_idx[p == 0]):
        pair0[(int(a), int(b))] = pair0.get((int(a), int(b)), 0) + 1
    return cut, cross0, inward, pair0


def _dangling_bonds(mask, terms):
    """Bonds of kept slab atoms to missing atoms at one surface when the
    period-0 copies of *mask* are deleted (see :func:`_dangling_terms`)."""
    cut, cross0, inward, pair0 = terms
    m = list(mask)
    ms = set(m)
    inside = sum(c for (a, b), c in pair0.items() if a in ms and b in ms)
    return int(cut - cross0[m].sum() + inward[m].sum() - inside)


def _tasker3_slab_moments(q, r, plane_idx, mask, L, atoms_per_fu, min_layers=1):
    """
    Exact charge and dipole of the Tasker III slab, from atom positions.

    The slab of ``lt`` repeat units holds the cell's atoms at heights
    ``r + m L`` (``m < lt``) plus the cut plane at ``r + lt L``, without the
    *mask* atoms of its bottom (``m = 0``) and top (``m = lt``) copies; *r*
    are heights above the bottom cut.  Returns ``(dipole, charge,
    dipole_per_fu, charge_per_fu)``: dipole (e·Å, about the mean height) and
    net charge of the ``min_layers`` slab, and the largest |dipole| and
    |charge| per formula unit over every thickness ``lt >= min_layers``.
    """
    q = np.asarray(q, dtype=float)
    r = np.asarray(r, dtype=float)
    P = np.asarray(plane_idx, dtype=int)
    M = np.asarray(list(mask), dtype=int)
    Nu, Qu, Du, Ru = len(q), q.sum(), q @ r, r.sum()
    NP, QP, DP, RP = len(P), q[P].sum(), q[P] @ r[P], r[P].sum()
    NM, QM, DM, RM = len(M), q[M].sum(), q[M] @ r[M], r[M].sum()

    def moments(lt):
        qz = lt * Du + L * Qu * lt * (lt - 1) / 2 + DP + lt * L * QP - 2 * DM - lt * L * QM
        z = lt * Ru + L * Nu * lt * (lt - 1) / 2 + RP + lt * L * NP - 2 * RM - lt * L * NM
        n = lt * Nu + NP - 2 * NM
        Q = lt * Qu + QP - 2 * QM
        mu = qz - Q * z / n
        k = n / atoms_per_fu
        return mu, Q, abs(mu) / k, abs(Q) / k

    lts = sorted({min_layers, min_layers + 1, min_layers + 3, 10 * min_layers})
    values = [moments(lt) for lt in lts]
    # Thick-slab limits (the quadratic terms of the dipole cancel).
    k_unit = Nu / atoms_per_fu
    slope = Du - Qu * Ru / Nu + L * (QP - QM - Qu * (NP - NM) / Nu)
    mu0, Q0 = values[0][0], values[0][1]
    dipole_per_fu = max([v[2] for v in values] + [abs(slope) / k_unit])
    charge_per_fu = max([v[3] for v in values] + [abs(Qu) / k_unit])
    return float(mu0), float(Q0), float(dipole_per_fu), float(charge_per_fu)


def _prefer_matches(prefer_plane, label, counts):
    """
    Whether a plane matches *prefer_plane* (str or list of str): an element
    symbol selects planes made only of that element, any other string is a
    plane label (:func:`plane_name_matches`).  Integers are ignored.
    """
    if prefer_plane is None:
        return False
    queries = [prefer_plane] if isinstance(prefer_plane, str) else list(prefer_plane)
    present = {Z for Z, c in counts.items() if c > 0}
    for query in queries:
        if not isinstance(query, str):
            continue
        if query in atomic_numbers:
            if present == {atomic_numbers[query]}:
                return True
        elif plane_name_matches(query, label):
            return True
    return False


def _compute_distribution_score(kept_indices, atoms_z_matrix, numbers, dists, bond_distances):
    """
    Score how well-distributed the remaining atoms are on the
    reconstructed surface plane (lower is better).

    Pairs not listed in *bond_distances* contribute their Coulomb energy
    ``q_i * q_j / d_ij`` (charges from *atoms_z_matrix*, minimum-image
    distances *dists* between all atoms of the cell): like charges are
    pushed apart, so a half-occupied anion plane prefers a checkerboard over
    rows, and opposite charges stay close.

    Pairs listed in *bond_distances* keep their explicit rule: ``None``
    (forbidden) adds a repulsive ``1/d_ij``; a float adds
    ``|d_ij - d_ref|``.

    Each kind of term is averaged over its pairs and the averages are
    summed.
    """
    if len(kept_indices) < 2:
        return 0.0

    bd_map = _parse_bond_distances_map(bond_distances)
    kept = list(kept_indices)
    charges = atoms_z_matrix[:, 2]

    sums = {"forbidden": 0.0, "target": 0.0, "coulomb": 0.0}
    counts = {"forbidden": 0, "target": 0, "coulomb": 0}
    for a, i in enumerate(kept):
        for j in kept[a + 1:]:
            zi, zj = int(numbers[i]), int(numbers[j])
            pair = (min(zi, zj), max(zi, zj))
            d_ij = max(float(dists[i, j]), 1e-12)
            if pair in bd_map and bd_map[pair] is None:
                kind, value = "forbidden", 1.0 / d_ij
            elif pair in bd_map:
                kind, value = "target", abs(d_ij - bd_map[pair])
            else:
                kind, value = "coulomb", charges[i] * charges[j] / d_ij
            sums[kind] += value
            counts[kind] += 1

    return float(sum(sums[k] / counts[k] for k in sums if counts[k]))


def find_tasker3_candidates(
    planes_sorted,
    atoms_z_matrix,
    reduced_counts,
    adj,
    L,
    surf_bulk=None,
    bond_distances=None,
    charge_tol=1e-3,
    verbose=None,
    prefer_plane=None,
    plane_names=None,
    dipole_tol=0.05,
    bulk_atoms=None,
    miller=None,
    bond_threshold=(0.85, 1.15),
    min_layers=1,
    max_masks=200000,
):
    """
    Enumerate and score Tasker III reconstruction candidates.

    For each plane in the unit cell the stoichiometric excess is computed
    and every way of deleting it (the same atoms from both surface copies
    of the plane) is scored:

    - **dipole and charge** of the slab, exactly from the atom positions
      after the deletions; a candidate is valid when its |dipole| and
      |charge| per formula unit stay within *dipole_tol* / *charge_tol* for
      every thickness of at least *min_layers* repeat units;
    - **bond score**: bulk bonds of the kept atoms that end at missing atoms
      (vacuum or deleted), counted per surface cell over periodic images of
      the true bulk lattice (needs *bulk_atoms* and *miller*);
    - **distribution score** of the kept atoms of the surface plane (see
      :func:`_compute_distribution_score`).

    Patterns related by a symmetry operation of the crystal that keeps the
    stacking direction (in-plane rotations and mirrors, translations,
    screw axes) give the same slab up to that operation; only one of each
    set is scored, and its ``multiplicity`` is the size of the set.  This
    also merges symmetry-equivalent planes of the cell.

    Parameters
    ----------
    planes_sorted : list of dict
        Planes sorted by z-centre (from :func:`identify_planes`).
    atoms_z_matrix : ndarray
        ``[Z, z, q]`` matrix.
    reduced_counts : dict
        Reduced bulk stoichiometry.
    adj : ndarray or None
        Unused; kept for backward compatibility (bonds are counted
        geometrically, see *bulk_atoms*).
    L : float
        Lattice-plane spacing (angstrom).
    surf_bulk : Atoms or None
        One-layer cell from :func:`build_surface` (needed for the bond and
        distribution scores).
    bond_distances : dict or None
        Per-pair reference distances (same format as
        :func:`build_adjacency_matrix`).
    charge_tol : float
        Largest |net charge| per formula unit (e) treated as neutral.
    verbose : bool or None
        Print candidate table.
    prefer_plane : str, list[str], or None
        Candidates on matching planes are sorted first: an element symbol
        matches planes made only of that element, other strings match plane
        labels (``"O4"``, ``"O4-recon"``, ``"IrO2"``).
    plane_names : list of str or None
        Plane labels from :func:`assign_plane_names`.
    dipole_tol : float
        Largest |dipole| per formula unit (e·Å) treated as zero.
    bulk_atoms : Atoms or None
        Bulk cell *surf_bulk* was built from; with *miller*, bonds follow
        the true bulk lattice (:func:`surface_bulk_cell`).  Without it the
        cell of *surf_bulk* is used, which is right only when its third
        vector is a bulk lattice vector.
    miller : tuple of int or None
        Miller index of *surf_bulk*.
    bond_threshold : tuple of float
        ``(lo, hi)`` scaling of the reference bond distances.
    min_layers : int
        Thinnest slab (repeat units) the candidate must be valid for.
    max_masks : int
        Largest number of deletion patterns (before symmetry reduction) to
        enumerate; above it a ``ValueError`` is raised instead of running
        for a very long time.

    Returns
    -------
    list of dict
        Candidates ranked by ``(prefer match, valid, bond_score,
        distribution_score)``, ties broken by label and mask so the order is
        deterministic; IDs in rank order.  Each dict contains
        ``cut_plane_idx``, ``recon_label`` (e.g. ``"O4-recon"``),
        ``deletion_mask``, ``net_dipole`` and ``total_charge`` (of the
        *min_layers* slab), ``dipole_per_fu`` and ``charge_per_fu`` (largest
        over thicknesses), ``is_neutral``, ``is_valid``, ``bond_score``
        (``broken_top + broken_bottom``), ``distribution_score``,
        ``multiplicity``, ``plane_counts`` and more.  Invalid candidates are kept so they can
        be inspected; callers that build slabs keep the valid ones.
    """
    n = len(planes_sorted)
    q = np.asarray(atoms_z_matrix[:, 2], dtype=float)
    z = np.asarray(atoms_z_matrix[:, 1], dtype=float)
    atoms_per_fu = float(sum(reduced_counts.values()))
    if plane_names is None:
        plane_names = [_formula_label(p["counts"]) for p in planes_sorted]

    bonds = None
    dists = None
    perms = None
    numbers = atoms_z_matrix[:, 0].astype(int)
    if surf_bulk is not None:
        cell = surface_bulk_cell(bulk_atoms, miller) if bulk_atoms is not None else surf_bulk.cell
        periodic = Atoms(numbers=surf_bulk.numbers, positions=surf_bulk.positions,
                         cell=cell, pbc=True)
        bi, bj, D = _bond_pairs(periodic, bond_threshold, bond_distances)
        bonds = (bi, bj, D[:, 2])
        dists = surf_bulk.get_all_distances(mic=True)
        perms = _stacking_symmetry(surf_bulk.numbers, surf_bulk.positions, cell)

    # Number of patterns before enumerating them: the count is combinatorial
    # in the size of the surface cell.
    plane_excess = []
    total = 0
    for i, plane in enumerate(planes_sorted):
        excess, k = _compute_plane_excess(plane["counts"], reduced_counts)
        if excess is None or all(v == 0 for v in excess.values()):
            continue
        n_masks = 1
        for Z, n_del in excess.items():
            n_masks *= comb(plane["counts"].get(Z, 0), n_del)
        plane_excess.append((i, excess, k))
        total += n_masks
    if total > max_masks:
        raise ValueError(
            f"{total} Tasker III deletion patterns to enumerate (max_masks={max_masks}); "
            "use a smaller surface cell, or raise max_masks if you mean it."
        )

    plane_of = np.full(len(atoms_z_matrix), -1)
    for i, plane in enumerate(planes_sorted):
        plane_of[plane["indices"]] = i

    def orbit(i, mask):
        """Symmetry images (plane, sorted mask) of a pattern, or None if a
        smaller image exists (so the pattern is not its set's representative)."""
        key = (i, tuple(sorted(int(a) for a in mask)))
        images = {key}
        for perm in perms or ():
            image = perm[list(mask)]
            other = (int(plane_of[image[0]]), tuple(sorted(int(a) for a in image)))
            if other < key:
                return None
            images.add(other)
        return images

    plane_masks = []
    for i, excess, k in plane_excess:
        masks = []
        for mask in _enumerate_deletion_masks(planes_sorted[i]["indices"], atoms_z_matrix, excess):
            images = orbit(i, mask)
            if images is not None:
                masks.append((mask, len(images)))
        if masks:
            plane_masks.append((i, excess, k, masks))

    candidates = []
    for i, excess, k, masks in plane_masks:
        plane = planes_sorted[i]
        label = f"{plane_names[i]}-recon"
        zbot, ztop = compute_cut_positions(planes_sorted, L, (i - 1) % n, i)
        r_bot = (z - zbot) % L
        span = (ztop - zbot) % L or L
        r_top = (span - r_bot) % L  # depth below the top cut
        if bonds is not None:
            terms_bot = _dangling_terms(r_bot, bonds[0], bonds[1], bonds[2], L)
            terms_top = _dangling_terms(r_top, bonds[0], bonds[1], -bonds[2], L)
        matches = _prefer_matches(prefer_plane, label, plane["counts"])
        plane_charge = float(q[plane["indices"]].sum())

        for mask, multiplicity in masks:
            mu, total_q, dipole_per_fu, charge_per_fu = _tasker3_slab_moments(
                q, r_bot, plane["indices"], mask, L, atoms_per_fu, min_layers
            )
            if bonds is not None:
                broken_bottom = _dangling_bonds(mask, terms_bot)
                broken_top = _dangling_bonds(mask, terms_top)
            else:
                broken_bottom = broken_top = 0
            kept = [idx for idx in plane["indices"] if idx not in set(mask)]
            dist_score = (
                _compute_distribution_score(kept, atoms_z_matrix, numbers, dists, bond_distances)
                if dists is not None else 0.0
            )
            is_neutral = charge_per_fu <= charge_tol
            candidates.append({
                "cut_plane_idx": i,
                "recon_label": label,
                "plane_z": plane["z_center"],
                "plane_counts": dict(plane["counts"]),
                "deletion_mask": tuple(int(m) for m in mask),
                "excess": {Z: v for Z, v in excess.items() if v > 0},
                "n_deleted": len(mask),
                "formula_units_kept": k,
                "bond_score": broken_top + broken_bottom,
                "broken_top": broken_top,
                "broken_bottom": broken_bottom,
                "net_dipole": mu,
                "abs_dipole": abs(mu),
                "dipole_per_fu": dipole_per_fu,
                "total_charge": total_q,
                "charge_per_fu": charge_per_fu,
                "is_neutral": is_neutral,
                "is_valid": is_neutral and dipole_per_fu <= dipole_tol,
                "q_recon": plane_charge - float(q[list(mask)].sum()),
                "distribution_score": dist_score,
                "multiplicity": multiplicity,
                "matches_prefer_plane": matches,
            })

    def _rank(c):
        return (
            not c["matches_prefer_plane"] if prefer_plane is not None else False,
            not c["is_valid"],
            0.0 if c["is_valid"] else c["dipole_per_fu"],
            c["bond_score"],
            round(c["distribution_score"], 8),
            c["recon_label"],
            c["deletion_mask"],
        )

    candidates.sort(key=_rank)

    if verbose:
        print(f"\nTasker III reconstruction candidates: {len(candidates)}")
        print(
            f"{'#':>4s}  {'plane':>12s}  {'del':>3s}  {'excess':<16s}  {'Q/fu':>8s}  "
            f"{'mu/fu':>10s}  {'brkn':>5s}  {'(top':>5s}  {'bot)':>5s}  {'distr':>8s}  valid"
        )
        for rank, c in enumerate(candidates):
            excess_str = ", ".join(
                f"{chemical_symbols[Z]}:{m}" for Z, m in c["excess"].items()
            )
            print(
                f"{rank:4d}  {c['recon_label']:>12s}  {c['n_deleted']:3d}  {excess_str:<16s}  "
                f"{c['charge_per_fu']:8.4f}  {c['dipole_per_fu']:10.4e}  "
                f"{c['bond_score']:5d}  {c['broken_top']:5d}  {c['broken_bottom']:5d}  "
                f"{c['distribution_score']:+8.4f}  {c['is_valid']}"
            )

    return candidates


def _select_tasker3_candidates(candidates, miller, dipole_tol, charge_tol):
    """
    Keep the Tasker III candidates that give neutral, non-polar slabs.

    Raises ``ValueError`` with an explanation when there are none.
    """
    if not candidates:
        raise ValueError(
            f"No Tasker III reconstruction candidates for {tuple(miller)}: in this "
            "cell no plane can be made stoichiometric by removing the same atoms "
            "from both surfaces (the excess per surface is an odd number of atoms). "
            "An in-plane supercell usually fixes this: pass surface_supercell=(2, 1) "
            "or (2, 2) (not bulk_atoms * (2, 2, 1), which changes the facet unless "
            "the surface normal is along c)."
        )
    valid = [c for c in candidates if c["is_neutral"] and c["dipole_per_fu"] <= dipole_tol]
    if valid:
        return valid
    neutral = [c for c in candidates if c["is_neutral"]]
    if not neutral:
        best = min(candidates, key=lambda c: c["charge_per_fu"])
        raise ValueError(
            f"No charge-neutral Tasker III reconstruction found for {tuple(miller)}: "
            f"the best leaves {best['charge_per_fu']:.4g} e per formula unit "
            f"(charge_tol={charge_tol}).  Check that the charges sum to zero over "
            "the bulk cell; computed charges may need a larger charge_tol."
        )
    best = min(neutral, key=lambda c: c["dipole_per_fu"])
    raise ValueError(
        f"No non-polar Tasker III reconstruction found for {tuple(miller)}: "
        "removing atoms symmetrically from one plane type leaves a dipole of "
        f"at least {best['dipole_per_fu']:.4g} e*A per formula unit "
        f"(dipole_tol={dipole_tol}). "
        "This stacking needs different reconstructions on the two surfaces, "
        "which taskerslabgen does not build."
    )


def _reconstruction_metadata(cand, planes_sorted, plane_names, plane_name_map,
                             atoms_z_matrix, surf_bulk, bulk_atoms, miller, L):
    """
    Description of a Tasker III pattern that :func:`cutslab` can re-apply to
    any copy of the cut plane (JSON-serialisable).

    Fractional positions refer to the in-plane cell ``cell2d`` of
    *surf_bulk*.  ``a3_frac`` is the in-plane part of the bulk vector that
    stacks one repeat unit onto the next, so the pattern on copy ``m`` of
    the plane is ``delete_info + m * a3_frac``, as in
    :func:`build_tasker3_slabs`.  ``neighbor_planes`` are the bulk planes
    directly below and above the cut plane, used to align the pattern.
    """
    n = len(planes_sorted)
    i = cand["cut_plane_idx"]
    cut_plane = planes_sorted[i]

    cell2d = np.array(surf_bulk.cell[:2, :2], dtype=float)
    a3 = surface_bulk_cell(bulk_atoms, miller)[2]
    a3_frac = np.asarray(a3[:2]) @ np.linalg.inv(cell2d)

    def plane_frac(plane, copy=0):
        """Atoms of *plane* as [Z, fx, fy]; *copy* repeat units up (a3)."""
        return [
            [Z, (fx + copy * a3_frac[0]) % 1.0, (fy + copy * a3_frac[1]) % 1.0]
            for Z, fx, fy in compute_delete_info(plane, plane["indices"], atoms_z_matrix, surf_bulk)
        ]

    delete_info = compute_delete_info(cut_plane, cand["deletion_mask"], atoms_z_matrix, surf_bulk)
    recon_counts = dict(cut_plane["counts"])
    for species, _, _ in delete_info:
        recon_counts[species] -= 1
    return {
        "cut_plane_name": plane_names[i],
        "recon_label": cand["recon_label"],
        "cut_plane_counts": dict(cut_plane["counts"]),
        "recon_counts": {Z: c for Z, c in recon_counts.items() if c > 0},
        "cut_plane_frac": plane_frac(cut_plane),
        "delete_info": [list(a) for a in delete_info],
        # The planes physically adjacent to the cut plane: across the cell
        # boundary they are the copies one repeat unit down / up.
        "neighbor_planes": {
            "below": plane_frac(planes_sorted[(i - 1) % n], -1 if i == 0 else 0),
            "above": plane_frac(planes_sorted[(i + 1) % n], 1 if i == n - 1 else 0),
        },
        "cell2d": cell2d.tolist(),
        "a3_frac": a3_frac.tolist(),
        "period": float(L),
        "plane_names": list(plane_names),
        "plane_name_map": plane_name_map,
    }


def build_tasker3_slabs(
    bulk_atoms,
    miller,
    layer_thickness_list,
    cut_plane_idx,
    deletion_mask,
    planes_sorted,
    atoms_z_matrix,
    L,
    vacuum=15.0,
    plane_tol=None,
):
    """
    Build Tasker III slabs with symmetric surface reconstruction.

    Cuts at the specified plane, then deletes the copies of the
    *deletion_mask* atoms from both the bottom and the top copy of that
    plane, so both surfaces carry the same (translation-related) pattern
    and the slab is stoichiometric.

    Parameters
    ----------
    bulk_atoms : Atoms
        Bulk unit cell.
    miller : tuple of int
        Miller index ``(h, k, l)``.
    layer_thickness_list : list of int
        Slab thicknesses in bulk repeat units.
    cut_plane_idx : int
        Index into *planes_sorted* identifying the reconstruction plane.
    deletion_mask : tuple
        Indices (into the 1-layer cell / *atoms_z_matrix*) of the atoms to
        delete from the reconstruction plane.
    planes_sorted : list of dict
        Planes sorted by z-centre.
    atoms_z_matrix : ndarray
        ``[Z, z, q]`` matrix from :func:`compute_projection`.
    L : float
        Lattice-plane spacing (angstrom).
    vacuum : float
        Vacuum to add (angstrom, applied to each side).
    plane_tol : float or None
        Unused; kept for backward compatibility.  The surface planes are
        located from the cut positions, not by a z tolerance.

    Returns
    -------
    list of Atoms
        One slab per requested thickness, sorted by atom count.
    """
    n_uc = len(atoms_z_matrix)
    n_planes = len(planes_sorted)

    # The cut plane is the only plane between these two cuts.
    zbot, ztop = compute_cut_positions(
        planes_sorted, L, (cut_plane_idx - 1) % n_planes, cut_plane_idx
    )
    span = (ztop - zbot) % L or L
    deleted = np.array(sorted(int(i) for i in deletion_mask), dtype=int)

    slabs = []
    for lt in layer_thickness_list:
        n_layers = lt + 3
        slab_full = build_surface(bulk_atoms, miller, layers=n_layers)
        if not np.array_equal(slab_full.numbers, np.tile(bulk_atoms.numbers, n_layers)):
            raise RuntimeError("Unexpected atom order from ase.build.surface.")
        uc_index = np.arange(len(slab_full)) % n_uc
        z = slab_full.positions[:, 2]

        keep = (z >= zbot) & (z <= zbot + lt * L + span)
        bottom = (z >= zbot) & (z <= zbot + span)
        top = (z >= zbot + lt * L) & (z <= zbot + lt * L + span)
        remove = (bottom | top) & np.isin(uc_index, deleted)

        slab = slab_full[keep & ~remove]
        slab.set_pbc((True, True, True))
        apply_vacuum_to_slab(slab, vacuum=vacuum, axis=2)
        slabs.append(slab)

    slabs.sort(key=len)
    return slabs


def reconstruct_tasker_iii(
    bulk_atoms,
    charges,
    miller,
    layer_thickness_list,
    bulk_name,
    plane_tol=None,
    charge_tol=1e-3,
    dipole_tol=0.05,
    vacuum=15.0,
    plot=False,
    plot_out_dir=".",
    verbose=None,
    bond_threshold=(0.85, 1.15),
    bond_distances=None,
    prefer_plane=None,
    surface_supercell=None,
    max_masks=200000,
):
    """
    Standalone Tasker III reconstruction pipeline.

    Can be called directly when you already know the surface is Tasker III,
    or it is called automatically by :func:`generate_slabs_for_miller`.

    Parameters
    ----------
    bulk_atoms : Atoms
        Bulk unit cell.
    charges : dict, list or None
        Formal charges (same format as :func:`generate_slabs_for_miller`).
    miller : tuple of int
        Miller index ``(h, k, l)``.
    layer_thickness_list : list of int
        Slab thicknesses in bulk repeat units.
    bulk_name : str
        Label used in filenames.
    plane_tol : float or None
        Largest z-gap (angstrom) within one plane.  ``None`` (default)
        uses 0.1 Å.
    charge_tol : float
        Largest |net charge| per formula unit (e) treated as neutral.
    dipole_tol : float
        Largest |dipole| per formula unit (e·Å) still treated as zero
        (default 0.05).  Genuinely polar repeat units are ~1-6 e·Å per
        formula unit; relaxed structures may need ~0.3.
    vacuum : float
        Vacuum to add (angstrom, per side).
    plot : bool
        Generate a stacking-axis plot (default ``False``).
    plot_out_dir : str
        Directory for plot files.
    verbose : bool or None
        Print detailed output.
    bond_threshold : tuple of float
        ``(lo, hi)`` scaling factors for bond detection.
    bond_distances : dict or None
        Per-pair bond reference distances.
    prefer_plane : str, list[str], or None
        Candidates on matching planes are ranked first (element symbol or
        plane label, e.g. ``"O"``, ``"O4"``, ``"O4-recon"``).
    surface_supercell : tuple of int or None
        ``(n1, n2)`` in-plane repetition of the surface cell (see
        :func:`generate_slabs_for_miller`).
    max_masks : int
        Largest number of deletion patterns to enumerate (see
        :func:`find_tasker3_candidates`).

    Returns
    -------
    dict
        Contains ``"slab_atoms"``, ``"best_candidate"``,
        ``"all_candidates"`` (including rejected polar ones),
        ``"tasker_type"``, and ``"plot"`` path.

    Raises
    ------
    ValueError
        If no candidate gives a neutral, non-polar slab.
    """
    from .plotting import plot_unitcell_atoms

    h, k, l = miller
    if verbose:
        print(f"\nTasker III reconstruction for {bulk_name} ({h},{k},{l})\n")

    charges_list = _charges_to_list(bulk_atoms, charges)
    bulk = _tag_atom_indices(bulk_atoms)
    out_miller = tuple(miller)
    if surface_supercell is not None:
        bulk = _oriented_bulk(bulk, miller, surface_supercell)
        miller = (0, 0, 1)
    surf_bulk = build_surface(bulk, miller, layers=1, verbose=verbose)
    atoms_z_matrix, L = compute_projection(
        bulk, surf_bulk, [charges_list[i] for i in surf_bulk.arrays[_INDEX_KEY]], miller,
        verbose=verbose,
    )
    planes = identify_planes(
        atoms_z_matrix, L, plane_tol=plane_tol, charge_tol=charge_tol
    )
    reduced_counts = compute_reduced_counts(atoms_z_matrix)
    planes_sorted = sorted(planes, key=lambda p: p["z_center"] % L)

    if verbose:
        adj = build_adjacency_matrix(
            surf_bulk, bond_threshold=bond_threshold, bond_distances=bond_distances,
            bulk_atoms=bulk, miller=miller,
        )
        print(f"Planes: {len(planes_sorted)}, reduced: {reduced_counts}, "
              f"bonds: {int(np.sum(adj)) // 2}\n")
        print_adjacency_matrix(adj, surf_bulk)

    plane_names, _ = assign_plane_names(planes_sorted, atoms=surf_bulk)
    candidates = find_tasker3_candidates(
        planes_sorted, atoms_z_matrix, reduced_counts, None, L,
        surf_bulk=surf_bulk, bond_distances=bond_distances,
        charge_tol=charge_tol, verbose=verbose,
        prefer_plane=prefer_plane,
        plane_names=plane_names, dipole_tol=dipole_tol,
        bulk_atoms=bulk, miller=miller, bond_threshold=bond_threshold,
        min_layers=min(layer_thickness_list), max_masks=max_masks,
    )
    best = _select_tasker3_candidates(candidates, out_miller, dipole_tol, charge_tol)[0]
    if verbose:
        print(
            f"\n→ Best: {best['recon_label']}  "
            f"mu={best['net_dipole']:+.4e}  bonds_broken={best['bond_score']}\n"
        )

    zbot, ztop = compute_cut_positions(
        planes_sorted, L, (best["cut_plane_idx"] - 1) % len(planes_sorted),
        best["cut_plane_idx"],
    )

    plot_path = None
    if plot:
        plot_path = f"{plot_out_dir}/{bulk_name}_hkl_{h}{k}{l}_tasker3.png"
        plot_unitcell_atoms(
            atoms_z_matrix, L, out_miller,
            out_png=plot_path, plane_tol=plane_tol, planes=planes,
            zbot=zbot, ztop=ztop, dipole=best["net_dipole"],
        )

    slabs = build_tasker3_slabs(
        bulk, miller, layer_thickness_list,
        cut_plane_idx=best["cut_plane_idx"],
        deletion_mask=best["deletion_mask"],
        planes_sorted=planes_sorted,
        atoms_z_matrix=atoms_z_matrix,
        L=L, vacuum=vacuum,
    )
    for slab in slabs:
        _finalize_slab(slab, charges_list, reduced_counts, charge_tol, dipole_tol)
        slab.info["bulk_name"] = bulk_name
        slab.info["miller"] = out_miller

    return {
        "plot": plot_path,
        "slab_atoms": slabs,
        "best_candidate": best,
        "all_candidates": candidates,
        "tasker_type": "III",
    }
