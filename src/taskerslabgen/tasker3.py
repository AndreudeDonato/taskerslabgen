import numpy as np
from itertools import combinations
from math import comb as math_comb

from ase import Atoms
from ase.data import atomic_numbers, covalent_radii, chemical_symbols
from ase.neighborlist import neighbor_list

from .core import (
    _INDEX_KEY,
    _charges_to_list,
    _finalize_slab,
    _tag_atom_indices,
    build_surface,
    compute_projection,
    identify_planes,
    compute_reduced_counts,
    assign_plane_names,
    apply_vacuum_to_slab,
    compute_cut_positions,
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


def _compute_broken_bonds(adj, deleted_indices, excluded_layer_indices):
    """
    Count bonds broken by deleting atoms from a surface plane.
    Bonds to the excluded layer (vacuum side) are not counted because
    that layer does not exist in the real slab.
    """
    excluded = set(excluded_layer_indices)
    deleted = set(deleted_indices)
    n = adj.shape[0]
    broken = 0
    for d in deleted:
        for j in range(n):
            if j in excluded or j in deleted:
                continue
            if adj[d, j]:
                broken += 1
    return broken


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


def _compute_distribution_score(
    kept_indices, atoms_z_matrix, surf_bulk, bond_distances,
):
    """
    Score how well-distributed the remaining atoms are on the
    reconstructed surface plane (lower is better).

    Pairs not listed in *bond_distances* contribute their Coulomb energy
    ``q_i * q_j / d_ij`` (charges from *atoms_z_matrix*, minimum-image
    distances): like charges are pushed apart, so a half-occupied anion
    plane prefers a checkerboard over rows, and opposite charges stay close.

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
    sub = surf_bulk[kept]
    sub.set_pbc((True, True, True))
    dists = sub.get_all_distances(mic=True)
    numbers = sub.numbers
    charges = atoms_z_matrix[kept, 2]
    n = len(sub)

    sums = {"forbidden": 0.0, "target": 0.0, "coulomb": 0.0}
    counts = {"forbidden": 0, "target": 0, "coulomb": 0}
    for ii in range(n):
        for jj in range(ii + 1, n):
            zi, zj = int(numbers[ii]), int(numbers[jj])
            pair = (min(zi, zj), max(zi, zj))
            d_ij = max(float(dists[ii, jj]), 1e-12)
            if pair in bd_map and bd_map[pair] is None:
                kind, value = "forbidden", 1.0 / d_ij
            elif pair in bd_map:
                kind, value = "target", abs(d_ij - bd_map[pair])
            else:
                kind, value = "coulomb", charges[ii] * charges[jj] / d_ij
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
):
    """
    Enumerate and score Tasker III reconstruction candidates.

    For each plane in the unit cell the stoichiometric excess is computed
    and all unique deletion masks are enumerated.  Each mask is scored by
    broken bonds, dipole moment, and surface charge distribution.  Masks
    that produce identical spatial deletion patterns are deduplicated.

    Parameters
    ----------
    planes_sorted : list of dict
        Planes sorted by z-centre (from :func:`identify_planes`).
    atoms_z_matrix : ndarray
        ``[Z, z, q]`` matrix.
    reduced_counts : dict
        Reduced bulk stoichiometry.
    adj : ndarray
        Boolean adjacency matrix.
    L : float
        Lattice-plane spacing (angstrom).
    surf_bulk : Atoms or None
        Surface slab used for distribution scoring.
    bond_distances : dict or None
        Per-pair reference distances (same format as
        :func:`build_adjacency_matrix`).
    charge_tol : float
        Tolerance for charge neutrality.
    verbose : bool or None
        Print candidate table.
    prefer_plane : str, list[str], or None
        Preferred plane element/type; matching candidates are sorted
        first.  Element matching is **exclusive** (``"O"`` matches only
        pure-O planes).
    plane_names : list of str or None
        Plane labels from :func:`assign_plane_names` (e.g. ``["O4", "Ce2", ...]``).

    Returns
    -------
    list of dict
        Candidates sorted by ``(prefer_match, abs_dipole, bond_score,
        distribution_score)``.  Each dict contains ``cut_plane_idx``,
        ``deletion_mask``, ``net_dipole``, ``bond_score``,
        ``distribution_score``, ``plane_counts``, ``is_neutral``
        (``|total_charge| <= charge_tol``), ``dipole_per_fu`` (|dipole|
        per formula unit of thick slabs), and more.  Polar candidates are
        kept so they can be inspected; callers that build slabs keep only
        neutral ones with ``dipole_per_fu <= dipole_tol``.
    """
    n = len(planes_sorted)
    candidates = []
    # Formula units per bulk repeat unit.  A slab of lt repeat units has
    # dipole lt * net_dipole, so net_dipole / fu_per_unit is its dipole per
    # formula unit for thick slabs (the strictest value).
    fu_per_unit = len(atoms_z_matrix) / sum(reduced_counts.values())

    frac_all = surf_bulk.get_scaled_positions() if surf_bulk is not None else None

    total_raw_combos = 0
    for i in range(n):
        plane = planes_sorted[i]
        excess, k = _compute_plane_excess(plane["counts"], reduced_counts)
        if excess is None or all(v == 0 for v in excess.values()):
            continue
        groups = {}
        for idx in plane["indices"]:
            Z = int(atoms_z_matrix[idx, 0])
            groups.setdefault(Z, []).append(idx)
        n_combos = 1
        for Z, n_del in excess.items():
            if n_del == 0:
                continue
            available = len(groups.get(Z, []))
            if available < n_del:
                n_combos = 0
                break
            n_combos *= math_comb(available, n_del)
        total_raw_combos += n_combos

    if total_raw_combos > 10000 and verbose:
        print(
            f"WARNING: ~{total_raw_combos} Tasker III deletion combinations "
            f"to evaluate. This may take a while."
        )

    for i in range(n):
        plane = planes_sorted[i]
        excess, k = _compute_plane_excess(plane["counts"], reduced_counts)

        if excess is None:
            continue
        if all(v == 0 for v in excess.values()):
            continue

        above = planes_sorted[(i + 1) % n]
        below = planes_sorted[(i - 1) % n]

        masks = _enumerate_deletion_masks(plane["indices"], atoms_z_matrix, excess)
        if not masks:
            continue

        if frac_all is not None and len(masks) > 1:
            seen_fingerprints = set()
            unique_masks = []
            for mask in masks:
                fp = frozenset(
                    (int(atoms_z_matrix[idx, 0]),
                     round(float(frac_all[idx, 0]) % 1.0, 3),
                     round(float(frac_all[idx, 1]) % 1.0, 3))
                    for idx in mask
                )
                if fp not in seen_fingerprints:
                    seen_fingerprints.add(fp)
                    unique_masks.append(mask)
            masks = unique_masks

        for mask in masks:
            deleted_list = list(mask)

            broken_top = _compute_broken_bonds(adj, deleted_list, above["indices"])
            broken_bottom = _compute_broken_bonds(adj, deleted_list, below["indices"])
            bond_score = broken_top + broken_bottom

            plane_charge = float(np.sum(atoms_z_matrix[plane["indices"], 2]))
            deleted_charges = float(np.sum(atoms_z_matrix[list(mask), 2]))
            q_recon = plane_charge - deleted_charges

            uc_charge = float(np.sum(atoms_z_matrix[:, 2]))
            total_q = uc_charge + plane_charge - 2 * deleted_charges

            z_P = plane["z_center"] % L
            z_center_slab = z_P + L / 2.0

            mu = 0.0
            for j_plane in range(n):
                if j_plane == i:
                    continue
                p_j = planes_sorted[j_plane]
                z_j = p_j["z_center"] % L
                if z_j < z_P:
                    z_j += L
                mu += p_j["q_total"] * (z_j - z_center_slab)

            kept_indices = [idx for idx in plane["indices"] if idx not in set(mask)]
            if surf_bulk is not None:
                dist_score = _compute_distribution_score(
                    kept_indices, atoms_z_matrix, surf_bulk, bond_distances,
                )
            else:
                dist_score = 0.0

            matches_prefer = False
            if prefer_plane is not None:
                # Elements of the cut plane, as in genslab's prefer_plane
                # filter (symmetric deletion removes at most half of each
                # element, so the reconstructed plane has the same ones).
                present_Zs = {Z for Z, c in plane["counts"].items() if c > 0}
                if isinstance(prefer_plane, str):
                    if plane_names is not None and any(
                        plane_name_matches(prefer_plane, n) for n in plane_names
                    ):
                        matches_prefer = plane_name_matches(
                            prefer_plane, plane_names[i]
                        )
                    else:
                        z = atomic_numbers.get(prefer_plane)
                        if z is not None:
                            matches_prefer = present_Zs == {z}
                else:
                    try:
                        for e in prefer_plane:
                            if isinstance(e, str) and plane_names is not None and any(
                                plane_name_matches(e, n) for n in set(plane_names)
                            ):
                                if plane_name_matches(e, plane_names[i]):
                                    matches_prefer = True
                                    break
                            else:
                                z = atomic_numbers[e] if isinstance(e, str) else int(e)
                                if present_Zs == {z}:
                                    matches_prefer = True
                                    break
                    except (TypeError, AttributeError, KeyError):
                        pass

            candidates.append({
                "cut_plane_idx": i,
                "plane_z": plane["z_center"],
                "plane_counts": dict(plane["counts"]),
                "deletion_mask": mask,
                "excess": {Z: v for Z, v in excess.items() if v > 0},
                "n_deleted": len(mask),
                "formula_units_kept": k,
                "bond_score": bond_score,
                "broken_top": broken_top,
                "broken_bottom": broken_bottom,
                "net_dipole": mu,
                "abs_dipole": abs(mu),
                "dipole_per_fu": abs(mu) / fu_per_unit,
                "total_charge": total_q,
                "is_neutral": abs(total_q) <= charge_tol,
                "q_recon": q_recon,
                "distribution_score": dist_score,
                "matches_prefer_plane": matches_prefer,
            })

    def _sort_key(c):
        base = (c["abs_dipole"], c["bond_score"], c["distribution_score"])
        if prefer_plane is not None:
            return (0 if c["matches_prefer_plane"] else 1,) + base
        return base

    candidates.sort(key=_sort_key)

    if verbose:
        print(f"\nTasker III reconstruction candidates: {len(candidates)}")
        print(
            f"{'#':>4s}  {'plane':>5s}  {'z':>7s}  {'del':>3s}  "
            f"{'excess':<16s}  {'Q_slab':>8s}  {'Q_surf':>8s}  "
            f"{'mu':>12s}  {'brkn':>5s}  {'(top':>5s}  {'bot)':>5s}  "
            f"{'distr':>8s}"
        )
        for i, c in enumerate(candidates):
            excess_str = ", ".join(
                f"{chemical_symbols[Z]}:{n}" for Z, n in c["excess"].items()
            )
            print(
                f"{i:4d}  {c['cut_plane_idx']:5d}  {c['plane_z']:7.3f}  "
                f"{c['n_deleted']:3d}  {excess_str:<16s}  "
                f"{c['total_charge']:+8.3f}  {c['q_recon']:+8.3f}  "
                f"{c['net_dipole']:+12.4e}  "
                f"{c['bond_score']:5d}  {c['broken_top']:5d}  {c['broken_bottom']:5d}  "
                f"{c['distribution_score']:+8.4f}"
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
            "An in-plane supercell, e.g. bulk_atoms * (2, 2, 1), usually fixes this."
        )
    valid = [
        c for c in candidates
        if c["dipole_per_fu"] <= dipole_tol and abs(c["total_charge"]) <= charge_tol
    ]
    if not valid:
        best = min(candidates, key=lambda c: c["dipole_per_fu"])
        raise ValueError(
            f"No non-polar Tasker III reconstruction found for {tuple(miller)}: "
            "removing atoms symmetrically from one plane type leaves a dipole of "
            f"at least {best['dipole_per_fu']:.4g} e*A per formula unit "
            f"(dipole_tol={dipole_tol}). "
            "This stacking needs different reconstructions on the two surfaces, "
            "which taskerslabgen does not build."
        )
    return valid


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
):
    """
    Standalone Tasker III reconstruction pipeline.

    Can be called directly when you already know the surface is Tasker III,
    or it is called automatically by :func:`generate_slabs_for_miller`.

    Parameters
    ----------
    bulk_atoms : Atoms
        Bulk unit cell.
    charges : dict or list
        Formal charges (same format as :func:`compute_projection`).
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
        Tolerance for charge neutrality.
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
        ``(lo, hi)`` scaling factors for adjacency matrix.
    bond_distances : dict or None
        Per-pair bond reference distances.
    prefer_plane : str, list[str], or None
        Preferred plane element/type for candidate ranking.

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

    adj = build_adjacency_matrix(
        surf_bulk, bond_threshold=bond_threshold, bond_distances=bond_distances,
        bulk_atoms=bulk, miller=miller,
    )
    if verbose:
        n_bonds = int(np.sum(adj)) // 2
        print(f"Planes: {len(planes_sorted)}, reduced: {reduced_counts}, bonds: {n_bonds}\n")
        print_adjacency_matrix(adj, surf_bulk)

    plane_names, _ = assign_plane_names(planes_sorted, atoms=surf_bulk)
    candidates = find_tasker3_candidates(
        planes_sorted, atoms_z_matrix, reduced_counts, adj, L,
        surf_bulk=surf_bulk, bond_distances=bond_distances,
        charge_tol=charge_tol, verbose=verbose,
        prefer_plane=prefer_plane,
        plane_names=plane_names,
    )
    best = _select_tasker3_candidates(candidates, miller, dipole_tol, charge_tol)[0]
    if verbose:
        print(
            f"\n→ Best: plane {best['cut_plane_idx']}  "
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
            atoms_z_matrix, L, miller,
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

    return {
        "plot": plot_path,
        "slab_atoms": slabs,
        "best_candidate": best,
        "all_candidates": candidates,
        "tasker_type": "III",
    }
