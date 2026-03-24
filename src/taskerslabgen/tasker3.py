import numpy as np
from itertools import combinations
from math import comb as math_comb

from ase.data import atomic_numbers, covalent_radii, chemical_symbols
from ase.build import surface

from .core import (
    build_surface,
    compute_projection,
    identify_planes,
    compute_reduced_counts,
    assign_plane_names,
    apply_vacuum_to_slab,
)


def build_adjacency_matrix(atoms, threshold=(0.85, 1.15), bond_distances=None,
                           bulk_atoms=None):
    """
    Build a boolean adjacency matrix using covalent radii and PBC.

    Parameters
    ----------
    atoms : Atoms
        The structure whose atom indices will label the adjacency matrix.
    bulk_atoms : Atoms, optional
        Original bulk cell.  When provided the **bulk** cell vectors are
        used for the minimum-image distance calculation, which avoids
        artifacts that arise when ``ase.build.surface`` produces a
        c-vector that is not a true bulk repeat along the surface normal.
    bond_distances : dict, optional
        Per-pair reference distances.
        Keys: ``"Ce-O"`` style strings (order irrelevant) or tuples.
        Values: ``float`` (scaled by *threshold*) or ``None`` (forbid).
    """
    from ase import Atoms as AseAtoms
    from ase.data import atomic_numbers as ase_atomic_numbers

    n = len(atoms)

    if bulk_atoms is not None:
        temp = AseAtoms(
            numbers=atoms.numbers,
            positions=atoms.get_positions(),
            cell=bulk_atoms.cell[:],
            pbc=[True, True, True],
        )
        dists = temp.get_all_distances(mic=True)
    else:
        dists = atoms.get_all_distances(mic=True)
    adj = np.zeros((n, n), dtype=bool)
    lo, hi = threshold

    manual_map = {}
    if bond_distances:
        for key, d in bond_distances.items():
            if isinstance(key, str):
                parts = key.split("-")
                if len(parts) != 2:
                    raise ValueError(f"Bond key must be 'X-Y', got: {key!r}")
                a_str, b_str = parts[0].strip(), parts[1].strip()
                a = ase_atomic_numbers[a_str]
                b = ase_atomic_numbers[b_str]
            else:
                a, b = key
                if isinstance(a, str):
                    a = ase_atomic_numbers[a]
                if isinstance(b, str):
                    b = ase_atomic_numbers[b]
            manual_map[(min(a, b), max(a, b))] = d

    numbers = atoms.numbers
    for i in range(n):
        for j in range(i + 1, n):
            zi, zj = int(numbers[i]), int(numbers[j])
            pair = (min(zi, zj), max(zi, zj))
            if pair in manual_map:
                ref = manual_map[pair]
                if ref is None:
                    continue
            else:
                ref = covalent_radii[zi] + covalent_radii[zj]
            if lo * ref <= dists[i, j] <= hi * ref:
                adj[i, j] = True
                adj[j, i] = True

    return adj


def print_adjacency_matrix(adj, atoms):
    """Print the adjacency matrix with element labels as row/column headers."""
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

    Forbidden pairs (None in bond_distances): should be far apart
    and evenly spaced -> penalty = -d_ij (large distance = good).

    Allowed pairs (float in bond_distances or covalent-radii fallback):
    should be close to the reference bond distance and evenly spaced ->
    penalty = |d_ij - d_ref|.

    Both terms are normalised by their pair count.
    """
    if len(kept_indices) < 2:
        return 0.0

    bd_map = _parse_bond_distances_map(bond_distances)

    sub = surf_bulk[list(kept_indices)]
    sub.set_pbc((True, True, True))
    dists = sub.get_all_distances(mic=True)
    numbers = sub.numbers
    n = len(sub)

    forbidden_sum = 0.0
    forbidden_count = 0
    allowed_sum = 0.0
    allowed_count = 0

    for ii in range(n):
        for jj in range(ii + 1, n):
            zi, zj = int(numbers[ii]), int(numbers[jj])
            pair = (min(zi, zj), max(zi, zj))
            d_ij = dists[ii, jj]

            if pair in bd_map:
                ref = bd_map[pair]
                if ref is None:
                    forbidden_sum += -d_ij
                    forbidden_count += 1
                else:
                    allowed_sum += abs(d_ij - ref)
                    allowed_count += 1
            else:
                ref = covalent_radii[zi] + covalent_radii[zj]
                allowed_sum += abs(d_ij - ref)
                allowed_count += 1

    score = 0.0
    if forbidden_count > 0:
        score += forbidden_sum / forbidden_count
    if allowed_count > 0:
        score += allowed_sum / allowed_count
    return score


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
    For each plane in the unit cell, compute stoichiometric excess,
    enumerate symmetric deletion masks, and score each by dipole and
    broken bonds.  The same mask is applied to both top and bottom
    surfaces of the slab.

    Masks that produce identical spatial deletion patterns (same species
    at the same fractional xy positions) are deduplicated before scoring.
    """
    n = len(planes_sorted)
    candidates = []

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

    if total_raw_combos > 10000:
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

            recon_counts = dict(plane["counts"])
            for idx in mask:
                Z = int(atoms_z_matrix[idx, 0])
                recon_counts[Z] = recon_counts.get(Z, 0) - 1

            matches_prefer = False
            if prefer_plane is not None:
                if isinstance(prefer_plane, str) and plane_names is not None:
                    matches_prefer = plane_names[i] == prefer_plane
                else:
                    try:
                        elements = set(prefer_plane)
                        for e in elements:
                            z = atomic_numbers[e] if isinstance(e, str) else int(e)
                            if recon_counts.get(z, 0) > 0:
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
                "total_charge": total_q,
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


def _midpoint(z_sorted, L, i):
    n = len(z_sorted)
    z0 = z_sorted[i]
    z1 = z_sorted[(i + 1) % n]
    if z1 < z0:
        z1 += L
    return 0.5 * (z0 + z1)


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
    plane_tol=0.05,
):
    """
    Build Tasker III slabs: cut at the given plane, then symmetrically
    delete the same atoms from both the top and bottom surface planes.

    Returns a list of Atoms objects (one per thickness).
    """
    cut_plane = planes_sorted[cut_plane_idx]
    n_planes = len(planes_sorted)
    z_sorted = np.array([p["z_center"] % L for p in planes_sorted])

    bot_cut_idx = (cut_plane_idx - 1) % n_planes
    top_cut_idx = cut_plane_idx
    zbot = _midpoint(z_sorted, L, bot_cut_idx)
    ztop = _midpoint(z_sorted, L, top_cut_idx)

    ref_slab = surface(bulk_atoms, miller, layers=1, vacuum=0.0)
    ref_slab.set_pbc((True, True, True))
    frac_ref = ref_slab.get_scaled_positions()

    deleted_set = set(deletion_mask)
    delete_info = []
    for idx in cut_plane["indices"]:
        if idx in deleted_set:
            species = int(atoms_z_matrix[idx, 0])
            fx = frac_ref[idx, 0] % 1.0
            fy = frac_ref[idx, 1] % 1.0
            delete_info.append((species, fx, fy))

    plane_z_uc = cut_plane["z_center"] % L

    slabs = []
    for lt in layer_thickness_list:
        slab_full = surface(bulk_atoms, miller, layers=lt + 4, vacuum=0.0)
        slab_full.set_pbc((True, True, True))

        span = (ztop - zbot) % L
        zmin = zbot
        zmax = zbot + lt * L + span

        keep_mask = [(zmin <= a.position[2] <= zmax) for a in slab_full]
        slab = slab_full[keep_mask]

        z_positions = slab.positions[:, 2]
        frac_slab = slab.get_scaled_positions()

        bottom_z = plane_z_uc
        if bottom_z < zmin:
            bottom_z += L * np.ceil((zmin - bottom_z) / L)
        top_z = bottom_z + lt * L

        bottom_plane_mask = np.abs(z_positions - bottom_z) < plane_tol
        top_plane_mask = np.abs(z_positions - top_z) < plane_tol

        bottom_indices = np.where(bottom_plane_mask)[0]
        top_indices = np.where(top_plane_mask)[0]

        to_delete = set()
        for target_indices in [bottom_indices, top_indices]:
            for species, fx, fy in delete_info:
                best_j = None
                best_dist = np.inf
                for j in target_indices:
                    if j in to_delete:
                        continue
                    if slab.numbers[j] != species:
                        continue
                    dfx = abs((frac_slab[j, 0] % 1.0) - fx)
                    dfy = abs((frac_slab[j, 1] % 1.0) - fy)
                    dfx = min(dfx, 1.0 - dfx)
                    dfy = min(dfy, 1.0 - dfy)
                    d = dfx**2 + dfy**2
                    if d < best_dist:
                        best_dist = d
                        best_j = j
                if best_j is not None:
                    to_delete.add(best_j)

        keep = [i for i in range(len(slab)) if i not in to_delete]
        slab = slab[keep]
        slab.set_pbc((True, True, True))
        apply_vacuum_to_slab(slab, vacuum=vacuum, axis=2)
        slabs.append(slab)

    return slabs


def reconstruct_tasker_iii(
    bulk_atoms,
    charges,
    miller,
    layer_thickness_list,
    bulk_name,
    plane_tol=0.1,
    charge_tol=1e-3,
    dipole_tol=1e-6,
    vacuum=15.0,
    plot=True,
    plot_out_dir=".",
    verbose=None,
    bond_threshold=(0.85, 1.15),
    bond_distances=None,
    prefer_plane=None,
):
    """
    Standalone Tasker III reconstruction pipeline.

    Can be called directly when you already know the surface is Tasker III,
    or it is called automatically from generate_slabs_for_miller.
    """
    from .plotting import plot_unitcell_atoms

    h, k, l = miller
    if verbose:
        print(f"\nTasker III reconstruction for {bulk_name} ({h},{k},{l})\n")

    surf_bulk = build_surface(
        bulk_atoms, miller, layers=1, vacuum=0.0, verbose=verbose
    )
    atoms_z_matrix, L = compute_projection(
        bulk_atoms, surf_bulk, charges, miller, verbose=verbose
    )
    planes = identify_planes(
        atoms_z_matrix, L, plane_tol=plane_tol, charge_tol=charge_tol
    )
    reduced_counts = compute_reduced_counts(atoms_z_matrix)
    planes_sorted = sorted(planes, key=lambda p: p["z_center"] % L)

    adj = build_adjacency_matrix(
        surf_bulk, threshold=bond_threshold, bond_distances=bond_distances,
        bulk_atoms=bulk_atoms,
    )
    if verbose:
        n_bonds = int(np.sum(adj)) // 2
        print(f"Planes: {len(planes_sorted)}, reduced: {reduced_counts}, bonds_broken: {n_bonds}\n")
        print_adjacency_matrix(adj, surf_bulk)

    plane_names, _ = assign_plane_names(planes_sorted)
    candidates = find_tasker3_candidates(
        planes_sorted, atoms_z_matrix, reduced_counts, adj, L,
        surf_bulk=surf_bulk, bond_distances=bond_distances,
        charge_tol=charge_tol, verbose=verbose,
        prefer_plane=prefer_plane,
        plane_names=plane_names,
    )
    if not candidates:
        raise ValueError("No Tasker III reconstruction candidates found.")

    best = candidates[0]
    if verbose:
        print(
            f"\n→ Best: plane {best['cut_plane_idx']}  "
            f"mu={best['net_dipole']:+.4e}  bonds_broken={best['bond_score']}\n"
        )

    z_s = np.array([p["z_center"] % L for p in planes_sorted])
    bot_idx = (best["cut_plane_idx"] - 1) % len(planes_sorted)
    top_idx = best["cut_plane_idx"]
    zbot = _midpoint(z_s, L, bot_idx)
    ztop = _midpoint(z_s, L, top_idx)

    plot_path = None
    if plot:
        plot_path = f"{plot_out_dir}/{bulk_name}_hkl_{h}{k}{l}_tasker3.png"
        plot_unitcell_atoms(
            atoms_z_matrix, L, miller,
            out_png=plot_path, plane_tol=plane_tol, planes=planes,
            zbot=zbot, ztop=ztop, dipole=best["net_dipole"],
        )

    slabs = build_tasker3_slabs(
        bulk_atoms, miller, layer_thickness_list,
        cut_plane_idx=best["cut_plane_idx"],
        deletion_mask=best["deletion_mask"],
        planes_sorted=planes_sorted,
        atoms_z_matrix=atoms_z_matrix,
        L=L, vacuum=vacuum, plane_tol=plane_tol,
    )

    return {
        "plot": plot_path,
        "slab_atoms": slabs,
        "best_candidate": best,
        "all_candidates": candidates,
        "tasker_type": "III",
    }
