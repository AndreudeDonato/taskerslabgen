import numpy as np
from ase.build import surface
from ase.data import atomic_numbers, chemical_symbols
from ase.io import read, write
from scipy.optimize import linear_sum_assignment


def build_surface(bulk_atoms, miller, layers=1, vacuum=0.0, verbose=None):
    slab = surface(bulk_atoms, miller, layers=layers, vacuum=vacuum)
    slab.set_pbc((True, True, True))
    if verbose:
        print("BULK")
        print(bulk_atoms, bulk_atoms.positions, "\n")
        print("REORIENTED BULK")
        print(slab, slab.positions, "\n")
    return slab


def compute_projection(bulk, surf_bulk, charges, miller, verbose=None):
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
        for Z in surf_bulk.numbers:
            if Z not in charge_map:
                raise ValueError(f"Missing charge for atomic number: {Z}")
            charges_list.append(charge_map[Z])
        charges = charges_list
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


def identify_planes(atoms_z, L, plane_tol=0.2, charge_tol=1e-3):
    if len(atoms_z) == 0:
        return []

    z_mod = atoms_z[:, 1] % L
    sort_idx = np.argsort(z_mod)
    planes = []
    current_indices = [sort_idx[0]]
    current_center = float(z_mod[sort_idx[0]])

    for idx in sort_idx[1:]:
        z = float(z_mod[idx])
        if abs(z - current_center) <= plane_tol:
            current_indices.append(idx)
            current_center = float(np.mean(z_mod[current_indices]))
        else:
            q_total = float(np.sum(atoms_z[current_indices, 2]))
            if abs(q_total) < charge_tol:
                q_total = 0.0
            counts = {}
            for Z in atoms_z[current_indices, 0].astype(int):
                counts[Z] = counts.get(Z, 0) + 1
            planes.append(
                {
                    "z_center": current_center,
                    "q_total": q_total,
                    "indices": current_indices,
                    "counts": counts,
                }
            )
            current_indices = [idx]
            current_center = z

    q_total = float(np.sum(atoms_z[current_indices, 2]))
    if abs(q_total) < charge_tol:
        q_total = 0.0
    counts = {}
    for Z in atoms_z[current_indices, 0].astype(int):
        counts[Z] = counts.get(Z, 0) + 1
    planes.append(
        {"z_center": current_center, "q_total": q_total, "indices": current_indices, "counts": counts}
    )

    if len(planes) > 1:
        first = planes[0]
        last = planes[-1]
        wrap_dist = (first["z_center"] + L) - last["z_center"]
        if abs(wrap_dist) <= plane_tol:
            merged_indices = last["indices"] + first["indices"]
            angles = (z_mod[merged_indices] / L) * 2.0 * np.pi
            sin_mean = np.mean(np.sin(angles))
            cos_mean = np.mean(np.cos(angles))
            merged_center = (np.arctan2(sin_mean, cos_mean) / (2.0 * np.pi)) * L
            if merged_center < 0.0:
                merged_center += L
            merged_q = float(np.sum(atoms_z[merged_indices, 2]))
            if abs(merged_q) < charge_tol:
                merged_q = 0.0
            merged_counts = {}
            for Z in atoms_z[merged_indices, 0].astype(int):
                merged_counts[Z] = merged_counts.get(Z, 0) + 1
            planes = (
                [
                    {
                        "z_center": merged_center,
                        "q_total": merged_q,
                        "indices": merged_indices,
                        "counts": merged_counts,
                    }
                ]
                + planes[1:-1]
            )
    return planes


def compute_reduced_counts(atoms_z):
    types = np.unique(atoms_z[:, 0].astype(int))
    counts = {Z: int(np.sum(atoms_z[:, 0] == Z)) for Z in types}
    gcd = 0
    for c in counts.values():
        gcd = np.gcd(gcd, c)
    gcd = max(int(gcd), 1)
    reduced = {Z: counts[Z] // gcd for Z in types}
    return reduced


def is_stoichiometric_sequence(sequence_counts, reduced_counts):
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
                }
            )


    sequences.sort(key=lambda s: abs(s["net_dipole"]), reverse=True)
    return sequences


def select_best_sequence(sequences, dipole_tol=1e-6):
    valid = [s for s in sequences if s["is_neutral"] and s["is_stoich"]]
    if not valid:
        return None
    best = valid[-1]
    best["is_tasker_ii"] = abs(best["net_dipole"]) <= dipole_tol
    return best


def compute_cut_positions(planes, L, bottom_cut_index, top_cut_index):
    planes_sorted = sorted(planes, key=lambda p: p["z_center"] % L)
    z_sorted = np.array([p["z_center"] % L for p in planes_sorted], dtype=float)
    n = len(z_sorted)

    def midpoint(i):
        z0 = z_sorted[i]
        z1 = z_sorted[(i + 1) % n]
        if z1 < z0:
            z1 += L
        return 0.5 * (z0 + z1)

    return midpoint(bottom_cut_index), midpoint(top_cut_index)


def assign_plane_names(planes_sorted):
    """
    Assign a name to each plane based on its elemental composition.

    Planes with the same composition get the same name (e.g. P0, P1).
    Returns (names, name_map) where names[i] is the name of planes_sorted[i]
    and name_map is {name: counts_dict}.
    """
    seen = {}
    counter = 0
    names = []
    name_map = {}
    for plane in planes_sorted:
        key = tuple(sorted(plane["counts"].items()))
        if key not in seen:
            name = f"P{counter}"
            seen[key] = name
            name_map[name] = dict(plane["counts"])
            counter += 1
        names.append(seen[key])
    return names, name_map


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


def extract_termination(reference, charges, axis=2, plane_tol=0.1, charge_tol=1e-3):
    """
    Extract termination fingerprints from a reference slab.

    Parameters
    ----------
    reference : Atoms, str/Path, or dict (genslab result)
        - Atoms / file: extract bottom/top fingerprints only.
        - Dict with ``"tasker_type"`` key: also extract reconstruction
          metadata so cutslab can reapply the same Tasker III pattern.
    charges : list or dict
        Charges per atom (list) or per element (dict).
    axis : int
        Stacking axis (default 2 = z).
    plane_tol, charge_tol : float
        Tolerances forwarded to identify_planes.

    Returns
    -------
    dict with keys:
        ``"bottom"``, ``"top"`` -- fingerprint dicts (counts + frac_xy).
        ``"reconstruction"``    -- None, or a dict with delete_info etc.
        ``"plane_names"``       -- None, or list of per-plane names.
        ``"plane_name_map"``    -- None, or {name: counts} mapping.
    """
    reconstruction = None
    plane_names = None
    plane_name_map = None

    if isinstance(reference, dict) and "tasker_type" in reference:
        slab = reference["slab_atoms"][0]
        if "reconstruction" in reference and reference["reconstruction"] is not None:
            reconstruction = reference["reconstruction"]
            plane_names = reconstruction["plane_names"]
            plane_name_map = reconstruction["plane_name_map"]
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

    Parameters
    ----------
    plane : dict
        Plane dict from identify_planes (must have 'counts' and 'indices').
    ref_fingerprint : dict
        One entry from extract_termination (has 'counts' and 'frac_xy').
    atoms : Atoms
        The structure the plane indices refer to (for reading positions).
    axis : int
        Stacking axis.

    Returns
    -------
    (matches, rmsd) : (bool, float)
        matches is False (rmsd = inf) when stoichiometry differs.
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


def generate_slabs_for_miller(
    bulk_atoms,
    charges,
    miller,
    layer_thickness_list,
    bulk_name,
    repeat=(1, 1, 1),
    out_dir=".",
    plot_out_dir=".",
    layers=1,
    plane_tol=0.1,
    charge_tol=1e-3,
    dipole_tol=1e-6,
    vacuum=10.0,
    plot=True,
    verbose=None,
    output_ext="xyz",
    bond_threshold=(0.85, 1.15),
    bond_distances=None,
):
    from .plotting import plot_unitcell_atoms
    from .builder import build_cut_slabs

    if verbose:
        h, k, l = miller
        print(f"\nGenerating Tasker slab for {bulk_name} with Miller index ({h}, {k}, {l})\n")

    surf_bulk = build_surface(bulk_atoms, miller, layers=layers, vacuum=0.0, verbose=verbose)
    atoms_z_matrix, L = compute_projection(
        bulk_atoms, surf_bulk, charges, miller, verbose=verbose
    )
    planes = identify_planes(atoms_z_matrix, L, plane_tol=plane_tol, charge_tol=charge_tol)
    reduced_counts = compute_reduced_counts(atoms_z_matrix)
    sequences = enumerate_cut_pairs(planes, L, reduced_counts, charge_tol=charge_tol)
    best_seq = select_best_sequence(sequences, dipole_tol=dipole_tol)
    if best_seq is None:
        raise ValueError("No valid stoichiometry sequences found.")

    if verbose:
        valid_sequences = [s for s in sequences if s["is_neutral"] and s["is_stoich"]]
        print("\nValid stoichiometry sequences (charge-neutral, reduced formula):")
        for i, seq in enumerate(valid_sequences):
            tasker_tag = "Tasker II" if abs(seq["net_dipole"]) <= dipole_tol else "Tasker III"
            bottom_edge = f"{seq['bottom_cut']}-{(seq['bottom_cut'] + 1) % len(planes)}"
            top_edge = f"{seq['top_cut']}-{(seq['top_cut'] + 1) % len(planes)}"
            print(
                f"{i:3d}  dir={seq['direction']}  "
                f"bottom_cut={bottom_edge} top_cut={top_edge}  "
                f"planes={seq['plane_indices']}  Q={seq['total_charge']:+.3f}  "
                f"mu={seq['net_dipole']:+.4e}  "
                f"z_center={seq['z_center']:.3f} {tasker_tag}"
            )

    h, k, l = miller

    # ---- Tasker I / II ----
    if best_seq["is_tasker_ii"]:
        bottom_cut_z, top_cut_z = compute_cut_positions(
            planes, L, best_seq["bottom_cut"], best_seq["top_cut"]
        )

        plot_path = None
        if plot:
            plot_path = f"{plot_out_dir}/{bulk_name}_hkl_{h}{k}{l}_atoms.png"
            plot_unitcell_atoms(
                atoms_z_matrix, L, miller,
                out_png=plot_path, plane_tol=plane_tol, planes=planes,
                zbot=bottom_cut_z, ztop=top_cut_z,
                dipole=best_seq["net_dipole"],
            )

        if output_ext is None:
            ext = None
            filename_template = None
        else:
            ext = output_ext.lstrip(".")
            filename_template = f"{bulk_name}_hkl_{h}{k}{l}_layers_{{layer_thickness}}.{ext}"
        slabs, slab_paths = build_cut_slabs(
            bulk_atoms=bulk_atoms, miller=miller,
            layer_thickness_list=layer_thickness_list,
            zbot=bottom_cut_z, ztop=top_cut_z, L=L,
            repeat=repeat, vacuum=vacuum,
            out_dir=out_dir, filename_template=filename_template, output_ext=ext,
        )

        return {
            "plot": plot_path,
            "slabs": slab_paths,
            "slab_atoms": slabs,
            "best_sequence": best_seq,
            "tasker_type": "I/II",
        }

    # ---- Tasker III: surface reconstruction ----
    if verbose:
        print("\nNo zero-dipole cut found → Tasker III surface reconstruction\n")

    from .tasker3 import (
        build_adjacency_matrix,
        find_tasker3_candidates,
        build_tasker3_slabs,
        print_adjacency_matrix,
        _midpoint,
    )

    planes_sorted = sorted(planes, key=lambda p: p["z_center"] % L)
    adj = build_adjacency_matrix(
        surf_bulk, threshold=bond_threshold, bond_distances=bond_distances,
    )
    if verbose:
        n_bonds = int(np.sum(adj)) // 2
        print(f"Adjacency: {n_bonds} bonds (threshold {bond_threshold})")
        print_adjacency_matrix(adj, surf_bulk)

    candidates = find_tasker3_candidates(
        planes_sorted, atoms_z_matrix, reduced_counts, adj, L,
        surf_bulk=surf_bulk, bond_distances=bond_distances,
        charge_tol=charge_tol, verbose=verbose,
    )
    if not candidates:
        raise ValueError("No Tasker III reconstruction candidates found.")

    best_t3 = candidates[0]

    plane_names, plane_name_map = assign_plane_names(planes_sorted)
    cut_plane = planes_sorted[best_t3["cut_plane_idx"]]
    cut_plane_name = plane_names[best_t3["cut_plane_idx"]]

    delete_info = compute_delete_info(
        cut_plane, best_t3["deletion_mask"], atoms_z_matrix, surf_bulk,
    )

    if verbose:
        recon_comp = dict(cut_plane["counts"])
        for sp, _, _ in delete_info:
            recon_comp[sp] = recon_comp.get(sp, 0) - 1
        recon_comp = {k: v for k, v in recon_comp.items() if v > 0}

        def _cl(c):
            return "+".join(
                f"{v}{chemical_symbols[k]}" if v > 1 else chemical_symbols[k]
                for k, v in sorted(c.items())
            )

        print(f"\nPlane types: {', '.join(f'{n}={_cl(plane_name_map[n])}' for n in dict.fromkeys(plane_names))}")
        print(
            f"→ Best Tasker III: {cut_plane_name}={_cl(cut_plane['counts'])}  "
            f"→  {cut_plane_name}-recon={_cl(recon_comp)}\n"
            f"  mu={best_t3['net_dipole']:+.4e}  bonds_broken={best_t3['bond_score']}\n"
        )

    z_s = np.array([p["z_center"] % L for p in planes_sorted])
    bot_idx = (best_t3["cut_plane_idx"] - 1) % len(planes_sorted)
    top_idx = best_t3["cut_plane_idx"]
    zbot = _midpoint(z_s, L, bot_idx)
    ztop = _midpoint(z_s, L, top_idx)

    recon_names = list(plane_names)
    recon_names[best_t3["cut_plane_idx"]] = f"{cut_plane_name}-recon"

    plot_path = None
    if plot:
        plot_path = f"{plot_out_dir}/{bulk_name}_hkl_{h}{k}{l}_tasker3.png"
        plot_unitcell_atoms(
            atoms_z_matrix, L, miller,
            out_png=plot_path, plane_tol=plane_tol, planes=planes,
            zbot=zbot, ztop=ztop, dipole=best_t3["net_dipole"],
            plane_names=recon_names,
        )

    if output_ext is None:
        ext = None
        fname_tmpl = None
    else:
        ext = output_ext.lstrip(".")
        fname_tmpl = f"{bulk_name}_hkl_{h}{k}{l}_tasker3_layers_{{layer_thickness}}.{ext}"

    slabs, slab_paths = build_tasker3_slabs(
        bulk_atoms, miller, layer_thickness_list,
        cut_plane_idx=best_t3["cut_plane_idx"],
        deletion_mask=best_t3["deletion_mask"],
        planes_sorted=planes_sorted,
        atoms_z_matrix=atoms_z_matrix,
        L=L,
        repeat=repeat, vacuum=vacuum,
        out_dir=out_dir, filename_template=fname_tmpl, output_ext=ext,
        plane_tol=plane_tol,
    )

    return {
        "plot": plot_path,
        "slabs": slab_paths,
        "slab_atoms": slabs,
        "best_candidate": best_t3,
        "all_candidates": candidates,
        "tasker_type": "III",
        "reconstruction": {
            "cut_plane_name": cut_plane_name,
            "cut_plane_counts": dict(cut_plane["counts"]),
            "delete_info": delete_info,
            "plane_names": plane_names,
            "plane_name_map": plane_name_map,
        },
    }


def cutslab(
    input_structure,
    charges,
    axis=2,
    plane_tol=0.1,
    charge_tol=1e-3,
    dipole_tol=1e-6,
    save_files=False,
    output_ext="xyz",
    out_dir=".",
    plot_out_dir=".",
    plot=True,
    verbose=None,
    bond_threshold=(0.85, 1.15),
    bond_distances=None,
    reference_termination=None,
    xy_tol=0.3,
    cuts="center",
):
    from .plotting import plot_unitcell_atoms

    if hasattr(input_structure, "positions"):
        atoms = input_structure.copy()
        stem = "structure"
    else:
        atoms = read(str(input_structure))
        stem = getattr(input_structure, "stem", None) or str(input_structure).split("/")[-1].split(".")[0]

    charges_list = _charges_to_list(atoms, charges)
    if len(charges_list) != len(atoms):
        raise ValueError(
            f"Charges length ({len(charges_list)}) does not match atoms ({len(atoms)})."
        )

    L = float(atoms.cell.lengths()[axis])
    if L <= 0.0:
        raise ValueError("Invalid cell length on selected axis.")

    coords = atoms.positions[:, axis]
    atoms_z_matrix = np.array(
        [[num, z, q] for num, z, q in zip(atoms.numbers, coords, charges_list)]
    )

    planes = identify_planes(atoms_z_matrix, L, plane_tol=plane_tol, charge_tol=charge_tol)
    reduced_counts = compute_reduced_counts(atoms_z_matrix)

    vec = [(1, 0, 0),
           (0, 1, 0),
           (0, 0, 1)]

    miller = vec[axis]

    planes_sorted = sorted(planes, key=lambda p: p["z_center"] % L)

    # ---- Reference-termination mode ----
    if reference_termination is not None:
        return _cutslab_reference(
            atoms, atoms_z_matrix, planes, planes_sorted, reduced_counts,
            L, axis, miller, charges, stem,
            reference_termination=reference_termination,
            xy_tol=xy_tol,
            plane_tol=plane_tol,
            charge_tol=charge_tol,
            dipole_tol=dipole_tol,
            save_files=save_files,
            output_ext=output_ext,
            out_dir=out_dir,
            plot_out_dir=plot_out_dir,
            plot=plot,
            verbose=verbose,
            cuts=cuts,
        )

    # ---- Standard mode (no reference) ----
    sequences = enumerate_cut_pairs(planes, L, reduced_counts, charge_tol=charge_tol)

    min_cut = max(seq["bottom_cut"] for seq in sequences)
    sequences.sort(key=lambda x: x['top_cut'])
    valid_sequences = [
        s for s in sequences
        if s["is_stoich"] and abs(s["net_dipole"]) <= dipole_tol and s["bottom_cut"] == min_cut
    ]

    # ---- Tasker I / II path ----
    if valid_sequences:
        if plot:
            plot_path = f"{plot_out_dir}/{stem}_axis_{axis}_cut.png"
            plot_unitcell_atoms(
                atoms_z_matrix, L, miller,
                out_png=plot_path, plane_tol=plane_tol, planes=planes,
                zbot=None, ztop=None, dipole=0,
            )

        cut_entries = []
        for seq in valid_sequences:
            zbot, ztop = compute_cut_positions(planes, L, seq["bottom_cut"], seq["top_cut"])
            cut_entries.append((zbot % L, seq, zbot, ztop))
        cut_entries.sort(key=lambda x: x[0])

        slab_atoms = []
        slab_paths = []
        ext = None if output_ext is None else output_ext.lstrip(".")

        for idx, (_, seq, zbot, ztop) in enumerate(cut_entries, start=1):
            atom_indices = []
            for pidx in seq["plane_indices"]:
                atom_indices.extend(planes_sorted[pidx]["indices"])
            atom_indices = sorted(set(atom_indices))
            slab = atoms[atom_indices]
            slab_atoms.append(slab)

            if save_files and ext is not None:
                out_path = f"{out_dir}/{stem}_axis_{axis}_cut_{idx}.{ext}"
                write(out_path, slab)
                slab_paths.append(out_path)

            if verbose:
                bottom_edge = f"{seq['bottom_cut']}-{(seq['bottom_cut'] + 1) % len(planes)}"
                top_edge = f"{seq['top_cut']}-{(seq['top_cut'] + 1) % len(planes)}"
                print(
                    f"{idx:3d}  bottom_cut={bottom_edge} top_cut={top_edge}  "
                    f"Q={seq['total_charge']:+.3f}  mu={seq['net_dipole']:+.4e}  "
                    f"z_center={seq['z_center']:.3f}"
                )

        return {
            "slab_atoms": slab_atoms,
            "slab_paths": slab_paths,
            "valid_sequences": [entry[1] for entry in cut_entries],
            "tasker_type": "I/II",
        }

    # ---- Tasker III path ----
    if verbose:
        print("\nNo zero-dipole cut found → Tasker III surface reconstruction\n")

    from .tasker3 import (
        build_adjacency_matrix,
        find_tasker3_candidates,
        print_adjacency_matrix,
    )

    adj = build_adjacency_matrix(
        atoms, threshold=bond_threshold, bond_distances=bond_distances,
    )
    if verbose:
        n_bonds = int(np.sum(adj)) // 2
        print(f"Adjacency: {n_bonds} bonds (threshold {bond_threshold})")
        print_adjacency_matrix(adj, atoms)

    candidates = find_tasker3_candidates(
        planes_sorted, atoms_z_matrix, reduced_counts, adj, L,
        surf_bulk=atoms, bond_distances=bond_distances,
        charge_tol=charge_tol, verbose=verbose,
    )
    if not candidates:
        raise ValueError("No Tasker III reconstruction candidates found.")

    best = candidates[0]
    if verbose:
        print(
            f"\n→ Best Tasker III: plane {best['cut_plane_idx']}  "
            f"mu={best['net_dipole']:+.4e}  bonds_broken={best['bond_score']}  "
            f"distr={best['distribution_score']:+.4f}\n"
        )

    if plot:
        from .tasker3 import _midpoint
        z_s = np.array([p["z_center"] % L for p in planes_sorted])
        bot_idx = (best["cut_plane_idx"] - 1) % len(planes_sorted)
        top_idx = best["cut_plane_idx"]
        zbot = _midpoint(z_s, L, bot_idx)
        ztop = _midpoint(z_s, L, top_idx)
        plot_path = f"{plot_out_dir}/{stem}_axis_{axis}_tasker3.png"
        plot_unitcell_atoms(
            atoms_z_matrix, L, miller,
            out_png=plot_path, plane_tol=plane_tol, planes=planes,
            zbot=zbot, ztop=ztop, dipole=best["net_dipole"],
        )

    slab_atoms = []
    slab_paths = []
    ext = None if output_ext is None else output_ext.lstrip(".")

    deleted_set = set(best["deletion_mask"])
    keep = [i for i in range(len(atoms)) if i not in deleted_set]
    slab = atoms[keep]
    slab_atoms.append(slab)

    if save_files and ext is not None:
        out_path = f"{out_dir}/{stem}_axis_{axis}_tasker3_cut_1.{ext}"
        write(out_path, slab)
        slab_paths.append(out_path)

    return {
        "slab_atoms": slab_atoms,
        "slab_paths": slab_paths,
        "best_candidate": best,
        "all_candidates": candidates,
        "tasker_type": "III",
    }


def _apply_reconstruction(slab, plane_indices, delete_info, axis=2):
    """
    Apply a Tasker III reconstruction pattern to a set of atoms
    identified as a surface plane.

    For each (species_Z, frac_x, frac_y) in delete_info, find the
    closest matching atom of that species in the plane and mark it
    for deletion.  No distance cutoff is applied because alternating
    planes in many crystal structures have shifted xy positions;
    the caller is responsible for verifying that the plane's
    composition matches the expected cut-plane composition.

    Returns list of atom indices (in slab) to delete.
    """
    frac = slab.get_scaled_positions()
    ab_axes = [i for i in range(3) if i != axis]
    to_delete = set()
    for species, fx, fy in delete_info:
        best_j = None
        best_d = np.inf
        for j in plane_indices:
            if j in to_delete:
                continue
            if slab.numbers[j] != species:
                continue
            dfx = abs((frac[j, ab_axes[0]] % 1.0) - fx)
            dfy = abs((frac[j, ab_axes[1]] % 1.0) - fy)
            dfx = min(dfx, 1.0 - dfx)
            dfy = min(dfy, 1.0 - dfy)
            d = np.sqrt(dfx**2 + dfy**2)
            if d < best_d:
                best_d = d
                best_j = j
        if best_j is not None:
            to_delete.add(best_j)
    return sorted(to_delete)


def _cutslab_reference(
    atoms, atoms_z_matrix, planes, planes_sorted, reduced_counts,
    L, axis, miller, charges, stem,
    reference_termination,
    xy_tol,
    plane_tol,
    charge_tol,
    dipole_tol,
    save_files,
    output_ext,
    out_dir,
    plot_out_dir,
    plot,
    verbose,
    cuts="center",
):
    """
    Reference-termination mode for cutslab.

    When the reference is a genslab result dict with Tasker III
    reconstruction info, plane names are propagated and the same
    reconstruction pattern is applied to every newly exposed surface.

    Parameters
    ----------
    cuts : str
        Direction of cuts.  ``"center"`` (default) shrinks the slab
        symmetrically from both sides; ``"left"`` fixes the top and
        moves the bottom upward; ``"right"`` fixes the bottom and
        moves the top downward; ``"all"`` returns every valid cut.
    """
    from .plotting import plot_unitcell_atoms

    ref_fp = extract_termination(
        reference_termination, charges, axis=axis,
        plane_tol=plane_tol, charge_tol=charge_tol,
    )
    ref_bot = ref_fp["bottom"]
    ref_top = ref_fp["top"]

    reconstruction = ref_fp.get("reconstruction")
    ref_plane_names = ref_fp.get("plane_names")
    ref_plane_name_map = ref_fp.get("plane_name_map")

    thick_plane_names, thick_name_map = assign_plane_names(planes_sorted)

    if ref_plane_name_map is not None:
        ref_comp_to_name = {}
        for name, counts in ref_plane_name_map.items():
            key = tuple(sorted(counts.items()))
            ref_comp_to_name[key] = name
        original_names = list(thick_plane_names)
        thick_plane_names = []
        for i, plane in enumerate(planes_sorted):
            key = tuple(sorted(plane["counts"].items()))
            if key in ref_comp_to_name:
                thick_plane_names.append(ref_comp_to_name[key])
            else:
                thick_plane_names.append(original_names[i])
        thick_name_map = ref_plane_name_map

    cut_plane_name = reconstruction["cut_plane_name"] if reconstruction else None
    cut_plane_counts = reconstruction["cut_plane_counts"] if reconstruction else None
    delete_info = reconstruction["delete_info"] if reconstruction else None

    n = len(planes_sorted)

    bot_scores = []
    top_scores = []
    for i, plane in enumerate(planes_sorted):
        b_match, b_rmsd = plane_match_score(plane, ref_bot, atoms, axis=axis)
        t_match, t_rmsd = plane_match_score(plane, ref_top, atoms, axis=axis)
        bot_scores.append((b_match and b_rmsd <= xy_tol, b_rmsd))
        top_scores.append((t_match and t_rmsd <= xy_tol, t_rmsd))

    bot_indices = [i for i in range(n) if bot_scores[i][0]]
    top_indices = [i for i in range(n) if top_scores[i][0]]
    matched_set = set(bot_indices) | set(top_indices)

    recon_eligible = set()
    if reconstruction and cut_plane_counts:
        for i, plane in enumerate(planes_sorted):
            if plane["counts"] == cut_plane_counts and i not in matched_set:
                recon_eligible.add(i)
        bot_indices = sorted(set(bot_indices) | recon_eligible)
        top_indices = sorted(set(top_indices) | recon_eligible)

    if verbose:
        def _comp_label(counts):
            parts = []
            for Z in sorted(counts):
                sym = chemical_symbols[Z]
                c = counts[Z]
                parts.append(f"{c}{sym}" if c > 1 else sym)
            return "+".join(parts)

        print(f"\nReference termination fingerprint:")
        print(f"  bottom: {_comp_label(ref_bot['counts'])}")
        print(f"  top:    {_comp_label(ref_top['counts'])}")
        if reconstruction:
            print(f"  reconstruction plane: {cut_plane_name} "
                  f"(unreconstructed={_comp_label(cut_plane_counts)})  "
                  f"delete_info: {len(delete_info)} atoms")
        print(f"\nPlane matching (xy_tol={xy_tol}):")
        for i, plane in enumerate(planes_sorted):
            b_ok, b_rmsd = bot_scores[i]
            t_ok, t_rmsd = top_scores[i]
            comp = _comp_label(plane["counts"])
            flags = []
            if b_ok:
                flags.append(f"BOT(rmsd={b_rmsd:.4f})")
            if t_ok:
                flags.append(f"TOP(rmsd={t_rmsd:.4f})")
            if i in recon_eligible:
                flags.append("RECON-ELIGIBLE")
            if i in matched_set:
                flags.append("ALREADY-RECON")
            tag = "  ".join(flags) if flags else "---"
            print(f"  {thick_plane_names[i]:>10s}  z={plane['z_center']:8.3f}  "
                  f"{comp:<16s}  {tag}")

    if not bot_indices:
        raise ValueError("No planes match the reference bottom termination.")
    if not top_indices:
        raise ValueError("No planes match the reference top termination.")

    z_arr = np.array([p["z_center"] for p in planes_sorted])
    q_arr = np.array([p["q_total"] for p in planes_sorted])

    recon_del_counts = {}
    recon_del_charge = 0.0
    if reconstruction and delete_info:
        charges_list = _charges_to_list(atoms, charges)
        charge_map = {}
        for Z_val in set(int(atoms.numbers[j]) for j in range(len(atoms))):
            for j in range(len(atoms)):
                if int(atoms.numbers[j]) == Z_val:
                    charge_map[Z_val] = charges_list[j]
                    break
        for species, _, _ in delete_info:
            recon_del_counts[species] = recon_del_counts.get(species, 0) + 1
            recon_del_charge += charge_map.get(species, 0.0)

    valid_cuts = []
    for bi in bot_indices:
        for ti in top_indices:
            if ti <= bi:
                continue

            seq_indices = list(range(bi, ti + 1))
            seq_counts = {}
            for pi in seq_indices:
                for Z, c in planes_sorted[pi]["counts"].items():
                    seq_counts[Z] = seq_counts.get(Z, 0) + c

            adj_counts = dict(seq_counts)
            adj_q_offset = 0.0
            n_recon_surfaces = 0
            if bi in recon_eligible:
                n_recon_surfaces += 1
            if ti in recon_eligible:
                n_recon_surfaces += 1
            for Z, nd in recon_del_counts.items():
                adj_counts[Z] = adj_counts.get(Z, 0) - nd * n_recon_surfaces
            adj_q_offset = recon_del_charge * n_recon_surfaces

            is_stoich, stoich_k = is_stoichiometric_sequence(adj_counts, reduced_counts)
            if not is_stoich:
                continue

            total_q = float(np.sum(q_arr[seq_indices])) - adj_q_offset
            if abs(total_q) > charge_tol:
                continue

            z_seq = z_arr[seq_indices]
            z_center = 0.5 * (z_seq[0] + z_seq[-1])
            q_adj = np.array(q_arr[seq_indices], dtype=float)
            if bi in recon_eligible:
                q_adj[0] -= recon_del_charge
            if ti in recon_eligible:
                q_adj[-1] -= recon_del_charge
            mu = float(np.sum(q_adj * (z_seq - z_center)))

            if abs(mu) > dipole_tol:
                continue

            valid_cuts.append({
                "bottom_plane": bi,
                "top_plane": ti,
                "plane_indices": seq_indices,
                "n_planes": len(seq_indices),
                "total_charge": total_q,
                "net_dipole": mu,
                "z_center": z_center,
                "stoich_k": stoich_k,
            })

    valid_cuts.sort(key=lambda c: c["n_planes"])

    if cuts != "all" and valid_cuts:
        all_bot = sorted(set(c["bottom_plane"] for c in valid_cuts))
        all_top = sorted(set(c["top_plane"] for c in valid_cuts), reverse=True)
        cut_lookup = {(c["bottom_plane"], c["top_plane"]): c for c in valid_cuts}

        if cuts == "center":
            selected = []
            n_pairs = min(len(all_bot), len(all_top))
            for k in range(n_pairs):
                pair = (all_bot[k], all_top[k])
                if pair[0] >= pair[1]:
                    break
                if pair in cut_lookup:
                    selected.append(cut_lookup[pair])
            valid_cuts = selected

        elif cuts == "left":
            fixed_top = all_top[0]
            valid_cuts = [c for c in valid_cuts if c["top_plane"] == fixed_top]

        elif cuts == "right":
            fixed_bot = all_bot[0]
            valid_cuts = [c for c in valid_cuts if c["bottom_plane"] == fixed_bot]

        else:
            raise ValueError(
                f"Unknown cuts mode: {cuts!r}. "
                f"Must be 'center', 'left', 'right', or 'all'."
            )

        valid_cuts.sort(key=lambda c: c["n_planes"], reverse=True)

    if verbose:
        print(f"\nValid reference-matched cuts (mode={cuts!r}): {len(valid_cuts)}")
        for i, cut in enumerate(valid_cuts):
            bn = thick_plane_names[cut["bottom_plane"]]
            tn = thick_plane_names[cut["top_plane"]]
            print(
                f"  {i:3d}  {bn}(P{cut['bottom_plane']})-{tn}(P{cut['top_plane']})  "
                f"({cut['n_planes']} planes)  "
                f"Q={cut['total_charge']:+.3f}  mu={cut['net_dipole']:+.4e}"
            )

    if not valid_cuts:
        raise ValueError(
            "No stoichiometric, charge-neutral, zero-dipole cuts found "
            "that match the reference termination."
        )

    plot_names = list(thick_plane_names)
    highlight_set = set(matched_set) | recon_eligible
    for i in range(n):
        if i in matched_set and reconstruction and thick_plane_names[i] == cut_plane_name:
            plot_names[i] = f"{cut_plane_name}-recon"

    if plot:
        plot_path = f"{plot_out_dir}/{stem}_axis_{axis}_refcut.png"
        plot_unitcell_atoms(
            atoms_z_matrix, L, miller,
            out_png=plot_path, plane_tol=plane_tol, planes=planes,
            zbot=None, ztop=None, dipole=0,
            matched_planes=highlight_set,
            plane_names=plot_names,
            title=f"cutslab reference-matched planes ({stem})",
        )

    slab_atoms = []
    slab_paths = []
    ext = None if output_ext is None else output_ext.lstrip(".")

    for idx, cut in enumerate(valid_cuts, start=1):
        atom_indices = []
        for pidx in cut["plane_indices"]:
            atom_indices.extend(planes_sorted[pidx]["indices"])
        atom_indices = sorted(set(atom_indices))
        slab = atoms[atom_indices]

        if reconstruction and delete_info:
            bi_plane = cut["bottom_plane"]
            ti_plane = cut["top_plane"]

            all_delete = []
            for pidx in [bi_plane, ti_plane]:
                if pidx not in recon_eligible:
                    continue
                plane_z = planes_sorted[pidx]["z_center"]
                surface_indices = [
                    j for j in range(len(slab))
                    if abs(slab.positions[j, axis] - plane_z) < plane_tol
                ]
                surface_counts = {}
                for j in surface_indices:
                    Z = int(slab.numbers[j])
                    surface_counts[Z] = surface_counts.get(Z, 0) + 1
                if surface_counts == cut_plane_counts:
                    del_idx = _apply_reconstruction(
                        slab, surface_indices, delete_info, axis=axis,
                    )
                    all_delete.extend(del_idx)
            if all_delete:
                keep = [i for i in range(len(slab)) if i not in set(all_delete)]
                slab = slab[keep]

        slab_atoms.append(slab)

        if save_files and ext is not None:
            out_path = f"{out_dir}/{stem}_axis_{axis}_refcut_{idx}.{ext}"
            write(out_path, slab)
            slab_paths.append(out_path)

    return {
        "slab_atoms": slab_atoms,
        "slab_paths": slab_paths,
        "valid_cuts": valid_cuts,
        "matched_planes": matched_set,
        "ref_fingerprint": ref_fp,
        "plane_names": thick_plane_names,
        "plane_name_map": thick_name_map,
        "tasker_type": "ref",
    }
