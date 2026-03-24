import numpy as np
from ase.data import chemical_symbols
from ase.io import read

from .core import (
    _charges_to_list,
    identify_planes,
    compute_reduced_counts,
    enumerate_cut_pairs,
    compute_cut_positions,
    apply_vacuum_to_slab,
    assign_plane_names,
    extract_termination,
    plane_match_score,
    is_stoichiometric_sequence,
)


def cutslab(
    input_structure,
    charges,
    axis=2,
    plane_tol=0.1,
    charge_tol=1e-3,
    dipole_tol=1e-6,
    plot_out_dir=".",
    plot=True,
    verbose=None,
    bond_threshold=(0.85, 1.15),
    bond_distances=None,
    reference_termination=None,
    cuts="center",
    vacuum=15.0,
    match_reference_planes=True,
    prefer_plane=None,
):
    """
    Cut an existing slab into thinner sub-slabs.

    Returns a list of Atoms objects (one per valid cut).

    Parameters
    ----------
    input_structure : Atoms or path
        The thick slab to cut.
    reference_termination : dict, Atoms, path, or None
        When provided, cutslab matches termination planes to the reference
        and applies the same reconstruction pattern if available.
    cuts : str
        ``"center"`` (default), ``"left"``, ``"right"``, or ``"all"``.
    """
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

    vec = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    miller = vec[axis]

    planes_sorted = sorted(planes, key=lambda p: p["z_center"] % L)

    # ---- Reference-termination mode ----
    if reference_termination is not None and match_reference_planes:
        return _cutslab_reference(
            atoms, atoms_z_matrix, planes, planes_sorted, reduced_counts,
            L, axis, miller, charges, stem,
            reference_termination=reference_termination,
            plane_tol=plane_tol,
            charge_tol=charge_tol,
            dipole_tol=dipole_tol,
            plot_out_dir=plot_out_dir,
            plot=plot,
            verbose=verbose,
            cuts=cuts,
            vacuum=vacuum,
        )

    # ---- Standard mode: cut wherever dipole is zero ----
    sequences = enumerate_cut_pairs(planes, L, reduced_counts, charge_tol=charge_tol)
    zero_dipole = [
        s for s in sequences
        if s["is_stoich"] and s["is_neutral"] and abs(s["net_dipole"]) <= dipole_tol
    ]

    if cuts == "all":
        valid_sequences = zero_dipole
    else:
        if not zero_dipole:
            valid_sequences = []
        else:
            all_bot = sorted(set(s["bottom_cut"] for s in zero_dipole))
            all_top = sorted(set(s["top_cut"] for s in zero_dipole), reverse=True)
            if cuts == "center":
                n_pairs = min(len(all_bot), len(all_top))
                valid_sequences = []
                for k in range(n_pairs):
                    pair = (all_bot[k], all_top[k])
                    if pair[0] >= pair[1]:
                        break
                    valid_sequences.extend(
                        s for s in zero_dipole
                        if s["bottom_cut"] == pair[0] and s["top_cut"] == pair[1]
                    )
            elif cuts == "left":
                fixed_top = all_top[0]
                valid_sequences = [s for s in zero_dipole if s["top_cut"] == fixed_top]
            elif cuts == "right":
                fixed_bot = all_bot[0]
                valid_sequences = [s for s in zero_dipole if s["bottom_cut"] == fixed_bot]
            else:
                raise ValueError(
                    f"Unknown cuts mode: {cuts!r}. "
                    f"Must be 'center', 'left', 'right', or 'all'."
                )

    valid_sequences.sort(key=lambda s: (s["bottom_cut"], s["top_cut"]))

    # ---- Tasker I/II path ----
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
        for idx, (_, seq, zbot, ztop) in enumerate(cut_entries, start=1):
            atom_indices = []
            for pidx in seq["plane_indices"]:
                atom_indices.extend(planes_sorted[pidx]["indices"])
            atom_indices = sorted(set(atom_indices))
            slab = atoms[atom_indices]
            apply_vacuum_to_slab(slab, vacuum=vacuum, axis=axis)
            slab_atoms.append(slab)

            if verbose:
                bottom_edge = f"{seq['bottom_cut']}-{(seq['bottom_cut'] + 1) % len(planes)}"
                top_edge = f"{seq['top_cut']}-{(seq['top_cut'] + 1) % len(planes)}"
                print(
                    f"{idx:3d}  bottom_cut={bottom_edge} top_cut={top_edge}  "
                    f"Q={seq['total_charge']:+.3f}  mu={seq['net_dipole']:+.4e}  "
                    f"z_center={seq['z_center']:.3f}"
                )

        return slab_atoms

    # ---- Tasker III path ----
    print(f"No Tasker I/II cut found for axis={axis}. Reconstructing Tasker III slab.")

    from .tasker3 import (
        build_adjacency_matrix,
        find_tasker3_candidates,
        print_adjacency_matrix,
        _midpoint,
    )

    plane_names, plane_name_map = assign_plane_names(planes_sorted)

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
        prefer_plane=prefer_plane,
        plane_names=plane_names,
    )
    if not candidates:
        raise ValueError("No Tasker III reconstruction candidates found.")

    best = candidates[0]
    if verbose:
        print(
            f"\n-> Best Tasker III: plane {best['cut_plane_idx']}  "
            f"mu={best['net_dipole']:+.4e}  bonds_broken={best['bond_score']}  "
            f"distr={best['distribution_score']:+.4f}\n"
        )

    if plot:
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

    deleted_set = set(best["deletion_mask"])
    keep = [i for i in range(len(atoms)) if i not in deleted_set]
    slab = atoms[keep]
    apply_vacuum_to_slab(slab, vacuum=vacuum, axis=axis)

    return [slab]


def _apply_reconstruction(slab, plane_indices, delete_info, axis=2):
    """
    Apply a Tasker III reconstruction pattern to a set of atoms
    identified as a surface plane.

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
    plane_tol,
    charge_tol,
    dipole_tol,
    plot_out_dir,
    plot,
    verbose,
    cuts="center",
    vacuum=15.0,
):
    """
    Reference-termination mode for cutslab.

    Returns a list of Atoms objects.
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
        bot_scores.append((b_match and b_rmsd <= plane_tol, b_rmsd))
        top_scores.append((t_match and t_rmsd <= plane_tol, t_rmsd))

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
        print(f"\nPlane matching (plane_tol={plane_tol}):")
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

        apply_vacuum_to_slab(slab, vacuum=vacuum, axis=axis)
        slab_atoms.append(slab)

    return slab_atoms
