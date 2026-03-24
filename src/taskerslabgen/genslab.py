import numpy as np
from ase.data import atomic_numbers, chemical_symbols
from ase.io import write

from .core import (
    build_surface,
    compute_projection,
    identify_planes,
    compute_reduced_counts,
    enumerate_cut_pairs,
    select_best_sequence,
    compute_cut_positions,
    assign_plane_names,
    compute_delete_info,
    is_stoichiometric_sequence,
)


def _filter_by_prefer_plane(terminations, prefer_plane):
    """
    Filter a {plane_id: info} dict by prefer_plane.

    prefer_plane semantics:
      - None      → keep only ID 0 (the best termination)
      - "all"     → keep every termination (no filtering)
      - int       → keep that single ID
      - list[int] → keep those IDs
      - str       → element filter (e.g. "O"): keep all terminations
                     whose cut plane contains any atoms of that element
      - list[str] → same, matching any of the listed elements
    """
    if prefer_plane == "all":
        return dict(terminations)

    if prefer_plane is None:
        if 0 in terminations:
            return {0: terminations[0]}
        return {}

    if isinstance(prefer_plane, int):
        prefer_plane = [prefer_plane]

    if isinstance(prefer_plane, (list, tuple)) and prefer_plane and all(isinstance(x, int) for x in prefer_plane):
        return {tid: terminations[tid] for tid in prefer_plane if tid in terminations}

    if isinstance(prefer_plane, str):
        elements = [prefer_plane]
    elif isinstance(prefer_plane, (list, tuple)) and prefer_plane and all(isinstance(x, str) for x in prefer_plane):
        elements = list(prefer_plane)
    else:
        raise ValueError(f"Invalid prefer_plane: {prefer_plane!r}")

    target_Zs = set()
    for e in elements:
        if e not in atomic_numbers:
            raise ValueError(f"Unknown element symbol: {e}")
        target_Zs.add(atomic_numbers[e])

    selected = {}
    for tid, term in terminations.items():
        counts = term.get("plane_counts", {})
        if any(Z in counts and counts[Z] > 0 for Z in target_Zs):
            selected[tid] = term

    if not selected:
        raise ValueError(f"No termination plane contains element(s) {elements}")
    return selected


def generate_slabs_for_miller(
    bulk_atoms,
    charges,
    millers,
    layer_thickness_list,
    bulk_name="slab",
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
    savecandidates=False,
    candidate_id=None,
):
    """
    Generate non-polar slabs for one or more Miller indices.

    Parameters
    ----------
    millers : tuple or list of tuples
        Single Miller index ``(h,k,l)`` or list of Miller indices.
    layer_thickness_list : list of int
        Slab thicknesses in bulk repeat units.
    prefer_plane : None, "all", int, list[int], str, or list[str]
        - ``None``: return only the best termination (ID 0).
        - ``"all"``: return every generated termination/reconstruction.
        - ``int`` or ``list[int]``: return terminations with those IDs.
        - ``str`` (element symbol, e.g. ``"O"``): return all terminations
          whose cut plane contains any atoms of that element.
    savecandidates : bool
        Save all valid candidates to an xyz file for visual inspection.
    candidate_id : int, optional
        Legacy shorthand for ``prefer_plane=[candidate_id]``.

    Returns
    -------
    dict
        ``{miller: {plane_id: {"atoms": [...], "tasker_type": ..., ...}}}``
    """
    if isinstance(millers, tuple) and len(millers) == 3 and all(isinstance(x, (int, float)) for x in millers):
        millers = [millers]

    result = {}
    for miller in millers:
        result[tuple(miller)] = _generate_for_one_miller(
            bulk_atoms, charges, tuple(miller), layer_thickness_list, bulk_name,
            plane_tol, charge_tol, dipole_tol, vacuum,
            plot, plot_out_dir, verbose,
            bond_threshold, bond_distances,
            prefer_plane, savecandidates, candidate_id,
        )

    return result


def _generate_for_one_miller(
    bulk_atoms, charges, miller, layer_thickness_list, bulk_name,
    plane_tol, charge_tol, dipole_tol, vacuum,
    plot, plot_out_dir, verbose,
    bond_threshold, bond_distances,
    prefer_plane, savecandidates, candidate_id,
):
    from .plotting import plot_unitcell_atoms
    from .builder import build_cut_slabs
    from .tasker3 import (
        build_adjacency_matrix,
        find_tasker3_candidates,
        build_tasker3_slabs,
        print_adjacency_matrix,
        _midpoint,
    )

    h, k, l = miller

    if verbose:
        print(f"\nGenerating Tasker slab for {bulk_name} with Miller index ({h}, {k}, {l})\n")

    surf_bulk = build_surface(bulk_atoms, miller, layers=1, vacuum=0.0, verbose=verbose)
    atoms_z_matrix, L = compute_projection(
        bulk_atoms, surf_bulk, charges, miller, verbose=verbose
    )
    planes = identify_planes(atoms_z_matrix, L, plane_tol=plane_tol, charge_tol=charge_tol)
    reduced_counts = compute_reduced_counts(atoms_z_matrix)
    sequences = enumerate_cut_pairs(planes, L, reduced_counts, charge_tol=charge_tol)
    best_seq = select_best_sequence(sequences, dipole_tol=dipole_tol)

    if best_seq is None:
        raise ValueError("No valid stoichiometry sequences found.")

    planes_sorted = sorted(planes, key=lambda p: p["z_center"] % L)
    plane_names, plane_name_map = assign_plane_names(planes_sorted)

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

    # ---- Tasker I/II ----
    if best_seq["is_tasker_ii"]:
        return _tasker12_path(
            bulk_atoms, charges, miller, layer_thickness_list, bulk_name,
            planes, planes_sorted, plane_names, plane_name_map,
            sequences, reduced_counts, atoms_z_matrix, L, surf_bulk,
            dipole_tol, vacuum, plane_tol,
            plot, plot_out_dir, verbose,
            prefer_plane, savecandidates, candidate_id,
        )

    # ---- Tasker III ----
    print(
        f"No Tasker I/II plane found for miller=({h},{k},{l}) "
        f"on {bulk_name}. Reconstructing Tasker III slab."
    )

    return _tasker3_path(
        bulk_atoms, charges, miller, layer_thickness_list, bulk_name,
        planes, planes_sorted, plane_names, plane_name_map,
        reduced_counts, atoms_z_matrix, L, surf_bulk,
        vacuum, plane_tol, charge_tol,
        plot, plot_out_dir, verbose,
        bond_threshold, bond_distances,
        prefer_plane, savecandidates, candidate_id,
    )


def _tasker12_path(
    bulk_atoms, charges, miller, layer_thickness_list, bulk_name,
    planes, planes_sorted, plane_names, plane_name_map,
    sequences, reduced_counts, atoms_z_matrix, L, surf_bulk,
    dipole_tol, vacuum, plane_tol,
    plot, plot_out_dir, verbose,
    prefer_plane, savecandidates, candidate_id,
):
    from .plotting import plot_unitcell_atoms
    from .builder import build_cut_slabs

    h, k, l = miller

    zero_dipole = [
        s for s in sequences
        if s["is_neutral"] and s["is_stoich"] and abs(s["net_dipole"]) <= dipole_tol
    ]

    all_terminations = {}
    for tid, seq in enumerate(zero_dipole):
        bot_plane_idx = (seq["bottom_cut"] + 1) % len(planes_sorted)
        all_terminations[tid] = {
            "sequence": seq,
            "plane_type": plane_names[bot_plane_idx],
            "plane_counts": dict(planes_sorted[bot_plane_idx]["counts"]),
        }

    if candidate_id is not None:
        if candidate_id not in all_terminations:
            raise ValueError(
                f"candidate_id={candidate_id} out of range [0, {len(all_terminations)-1}]"
            )
        selected = {candidate_id: all_terminations[candidate_id]}
    else:
        selected = _filter_by_prefer_plane(all_terminations, prefer_plane)

    if savecandidates and zero_dipole:
        from pathlib import Path
        out_p = Path(plot_out_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        all_slabs_for_xyz = []
        for tid, seq in enumerate(zero_dipole):
            zbot_i, ztop_i = compute_cut_positions(planes, L, seq["bottom_cut"], seq["top_cut"])
            slabs_i = build_cut_slabs(bulk_atoms, miller, [layer_thickness_list[0]], zbot_i, ztop_i, L, vacuum)
            slab = slabs_i[0].copy()
            for key in list(slab.arrays):
                if key not in ("positions", "numbers"):
                    del slab.arrays[key]
            slab.info = {"candidate_id": tid}
            all_slabs_for_xyz.append(slab)
        cand_path = out_p / f"{bulk_name}_hkl_{h}{k}{l}_candidates.xyz"
        write(cand_path.as_posix(), all_slabs_for_xyz, format="extxyz")
        if verbose:
            print(f"Saved {len(all_slabs_for_xyz)} Tasker I/II candidates to {cand_path}\n")

    if plot:
        first_tid = min(selected.keys()) if selected else 0
        first_seq = all_terminations.get(first_tid, all_terminations[0])["sequence"]
        zbot_plot, ztop_plot = compute_cut_positions(planes, L, first_seq["bottom_cut"], first_seq["top_cut"])
        plot_path = f"{plot_out_dir}/{bulk_name}_hkl_{h}{k}{l}_atoms.png"
        plot_unitcell_atoms(
            atoms_z_matrix, L, miller,
            out_png=plot_path, plane_tol=plane_tol, planes=planes,
            zbot=zbot_plot, ztop=ztop_plot, dipole=first_seq["net_dipole"],
            plane_names=plane_names,
        )

    output = {}
    for tid, term_info in selected.items():
        seq = term_info["sequence"]
        zbot, ztop = compute_cut_positions(planes, L, seq["bottom_cut"], seq["top_cut"])
        slabs = build_cut_slabs(bulk_atoms, miller, layer_thickness_list, zbot, ztop, L, vacuum)
        output[tid] = {
            "atoms": slabs,
            "tasker_type": "I/II",
            "plane_type": term_info["plane_type"],
            "plane_counts": term_info["plane_counts"],
            "reconstruction": None,
            "candidate": seq,
        }

    return output


def _tasker3_path(
    bulk_atoms, charges, miller, layer_thickness_list, bulk_name,
    planes, planes_sorted, plane_names, plane_name_map,
    reduced_counts, atoms_z_matrix, L, surf_bulk,
    vacuum, plane_tol, charge_tol,
    plot, plot_out_dir, verbose,
    bond_threshold, bond_distances,
    prefer_plane, savecandidates, candidate_id,
):
    from .plotting import plot_unitcell_atoms
    from .tasker3 import (
        build_adjacency_matrix,
        find_tasker3_candidates,
        build_tasker3_slabs,
        print_adjacency_matrix,
        _midpoint,
    )

    h, k, l = miller

    adj = build_adjacency_matrix(
        surf_bulk, threshold=bond_threshold, bond_distances=bond_distances,
        bulk_atoms=bulk_atoms,
    )
    if verbose:
        n_bonds = int(np.sum(adj)) // 2
        print(f"Adjacency: {n_bonds} bonds_broken (threshold {bond_threshold})")
        print_adjacency_matrix(adj, surf_bulk)

    t3_prefer = None
    if isinstance(prefer_plane, str) and prefer_plane != "all":
        t3_prefer = prefer_plane
    elif isinstance(prefer_plane, (list, tuple)) and prefer_plane and all(isinstance(x, str) for x in prefer_plane):
        t3_prefer = prefer_plane

    candidates = find_tasker3_candidates(
        planes_sorted, atoms_z_matrix, reduced_counts, adj, L,
        surf_bulk=surf_bulk, bond_distances=bond_distances,
        charge_tol=charge_tol, verbose=verbose,
        prefer_plane=t3_prefer,
        plane_names=plane_names,
    )
    if not candidates:
        raise ValueError("No Tasker III reconstruction candidates found.")

    all_terminations = {}
    for tid, cand in enumerate(candidates):
        all_terminations[tid] = {
            "candidate": cand,
            "plane_type": plane_names[cand["cut_plane_idx"]],
            "plane_counts": dict(cand["plane_counts"]),
        }

    if candidate_id is not None:
        if candidate_id not in all_terminations:
            raise ValueError(
                f"candidate_id={candidate_id} out of range [0, {len(all_terminations)-1}]"
            )
        selected = {candidate_id: all_terminations[candidate_id]}
    else:
        selected = _filter_by_prefer_plane(all_terminations, prefer_plane)

    if savecandidates:
        from pathlib import Path
        out_p = Path(plot_out_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        all_slabs_for_xyz = []
        for tid, cand in enumerate(candidates):
            slabs_i = build_tasker3_slabs(
                bulk_atoms, miller, [layer_thickness_list[0]],
                cut_plane_idx=cand["cut_plane_idx"],
                deletion_mask=cand["deletion_mask"],
                planes_sorted=planes_sorted,
                atoms_z_matrix=atoms_z_matrix,
                L=L, vacuum=vacuum, plane_tol=plane_tol,
            )
            slab = slabs_i[0].copy()
            for key in list(slab.arrays):
                if key not in ("positions", "numbers"):
                    del slab.arrays[key]
            slab.info = {"candidate_id": tid}
            all_slabs_for_xyz.append(slab)
        cand_path = out_p / f"{bulk_name}_hkl_{h}{k}{l}_candidates.xyz"
        write(cand_path.as_posix(), all_slabs_for_xyz, format="extxyz")
        if verbose:
            print(f"Saved {len(all_slabs_for_xyz)} Tasker III candidates to {cand_path}\n")

    output = {}
    for tid, term_info in selected.items():
        cand = term_info["candidate"]
        cut_plane = planes_sorted[cand["cut_plane_idx"]]
        cut_plane_name = plane_names[cand["cut_plane_idx"]]

        delete_info = compute_delete_info(
            cut_plane, cand["deletion_mask"], atoms_z_matrix, surf_bulk,
        )

        slabs = build_tasker3_slabs(
            bulk_atoms, miller, layer_thickness_list,
            cut_plane_idx=cand["cut_plane_idx"],
            deletion_mask=cand["deletion_mask"],
            planes_sorted=planes_sorted,
            atoms_z_matrix=atoms_z_matrix,
            L=L, vacuum=vacuum, plane_tol=plane_tol,
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

            print(
                f"  Termination {tid}: {cut_plane_name}={_cl(cut_plane['counts'])}  "
                f"-> {cut_plane_name}-recon={_cl(recon_comp)}  "
                f"mu={cand['net_dipole']:+.4e}  bonds_broken={cand['bond_score']}"
            )

        output[tid] = {
            "atoms": slabs,
            "tasker_type": "III",
            "plane_type": f"{cut_plane_name}-recon",
            "plane_counts": dict(cut_plane["counts"]),
            "reconstruction": {
                "cut_plane_name": cut_plane_name,
                "cut_plane_counts": dict(cut_plane["counts"]),
                "delete_info": delete_info,
                "plane_names": plane_names,
                "plane_name_map": plane_name_map,
            },
            "candidate": cand,
        }

    if plot and selected:
        first_tid = min(selected.keys())
        first_cand = selected[first_tid]["candidate"]

        z_s = np.array([p["z_center"] % L for p in planes_sorted])
        bot_idx = (first_cand["cut_plane_idx"] - 1) % len(planes_sorted)
        top_idx = first_cand["cut_plane_idx"]
        zbot = _midpoint(z_s, L, bot_idx)
        ztop = _midpoint(z_s, L, top_idx)

        recon_names = list(plane_names)
        recon_names[first_cand["cut_plane_idx"]] = f"{plane_names[first_cand['cut_plane_idx']]}-recon"

        plot_path = f"{plot_out_dir}/{bulk_name}_hkl_{h}{k}{l}_tasker3.png"
        plot_unitcell_atoms(
            atoms_z_matrix, L, miller,
            out_png=plot_path, plane_tol=plane_tol, planes=planes,
            zbot=zbot, ztop=ztop, dipole=first_cand["net_dipole"],
            plane_names=recon_names,
        )

    return output
