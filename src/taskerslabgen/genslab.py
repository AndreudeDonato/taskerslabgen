import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, chemical_symbols
from ase.io import write

from .core import (
    _charges_to_list,
    _finalize_slab,
    _max_z_gap,
    _tag_atom_indices,
    build_surface,
    compute_projection,
    identify_planes,
    compute_reduced_counts,
    enumerate_cut_pairs,
    select_best_sequence,
    compute_cut_positions,
    assign_plane_names,
    compute_delete_info,
    plane_name_matches,
    surface_bulk_cell,
)


def _filter_by_prefer_plane(terminations, prefer_plane):
    """
    Filter a {plane_id: info} dict by prefer_plane.

    prefer_plane semantics:
      - ``None``        → no filter (keep everything)
      - ``int``         → keep that single termination ID
      - ``list[int]``   → keep those termination IDs
      - ``str``         → element symbol (e.g. ``"O"``) or plane label
                           (e.g. ``"IrO2"`` matches ``IrO2-a``/``IrO2-b``/
                           ``IrO2-a-recon``; ``"IrO2-a"`` matches ``IrO2-a``
                           and ``IrO2-a-recon``).
      - ``list[str]``   → match any of the listed strings

    Element matching is **exclusive**: ``"O"`` keeps only planes whose
    atoms are *all* oxygen.  A mixed CeO plane would NOT match ``"O"``.
    Use ``["O", "Ce"]`` to keep pure-O planes OR pure-Ce planes (but
    still not mixed CeO planes).  To select mixed planes use the plane
    label (e.g. ``"Ce2O4"``).
    """
    if prefer_plane is None:
        return dict(terminations)

    if isinstance(prefer_plane, int):
        prefer_plane = [prefer_plane]

    if isinstance(prefer_plane, (list, tuple)) and prefer_plane and all(isinstance(x, int) for x in prefer_plane):
        return {tid: terminations[tid] for tid in prefer_plane if tid in terminations}

    if isinstance(prefer_plane, str):
        str_list = [prefer_plane]
    elif isinstance(prefer_plane, (list, tuple)) and prefer_plane and all(isinstance(x, str) for x in prefer_plane):
        str_list = list(prefer_plane)
    else:
        raise ValueError(f"Invalid prefer_plane: {prefer_plane!r}")

    element_Zs = set()
    plane_type_names = set()
    for s in str_list:
        if s in atomic_numbers:
            element_Zs.add(atomic_numbers[s])
        else:
            plane_type_names.add(s)

    selected = {}
    for tid, term in terminations.items():
        counts = term.get("plane_counts", {})
        present_Zs = {Z for Z, c in counts.items() if c > 0}

        if element_Zs:
            for target_Z in element_Zs:
                if present_Zs == {target_Z}:
                    selected[tid] = term
                    break
            if tid in selected:
                continue

        if plane_type_names:
            pt = term.get("plane_type", "")
            if any(plane_name_matches(q, pt) for q in plane_type_names):
                selected[tid] = term
                continue

    if not selected:
        raise ValueError(
            f"No termination matches prefer_plane={prefer_plane!r}. "
            f"Available plane labels: "
            f"{sorted(set(t.get('plane_type','') for t in terminations.values()))}"
        )
    return selected


def generate_slabs_for_miller(
    bulk_atoms,
    charges,
    millers,
    layer_thickness_list,
    bulk_name="slab",
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
    candidates="best",
    savecandidates=False,
):
    """
    Generate non-polar slabs for one or more Miller indices.

    Automatically classifies each surface as Tasker I/II (zero-dipole)
    or Tasker III (requires reconstruction) and returns fully built
    slab structures.

    Parameters
    ----------
    bulk_atoms : Atoms
        Bulk unit cell.
    charges : dict or list
        Formal charges.  A dict maps element symbols (or atomic numbers)
        to charge values; a list gives per-atom charges.
    millers : tuple or list of tuples
        Single Miller index ``(h, k, l)`` or list of Miller indices.
    layer_thickness_list : list of int
        Slab thicknesses in bulk repeat units.
    bulk_name : str
        Label used in plot and output filenames.
    plane_tol : float or None
        Largest z-gap (angstrom) between neighbouring atoms of one plane
        (single-linkage clustering).  ``None`` (default) uses 0.1 Å.
    charge_tol : float
        Tolerance for charge neutrality.
    dipole_tol : float
        Largest |dipole| per formula unit (e·Å) still treated as zero
        (default 0.05).  Genuinely polar repeat units are ~1-6 e·Å per
        formula unit; relaxed structures may need ~0.3.
        Used to classify Tasker I/II and to accept Tasker III
        reconstructions.
    vacuum : float
        Vacuum to add (angstrom, per side).
    plot : bool
        Generate stacking-axis plots (default ``False``).
    plot_out_dir : str
        Directory for output plots.
    verbose : bool or None
        Print detailed information.
    bond_threshold : tuple of float
        ``(lo, hi)`` scaling factors for bond detection: Tasker III bond
        scores and the broken-bond ranking of Tasker I/II terminations.
    bond_distances : dict or None
        Per-pair bond reference distances.  Keys are ``"X-Y"`` strings
        (e.g. ``"Ce-O"``).  Values are ``float`` (reference distance)
        or ``None`` (forbid that pair).
    prefer_plane : None, int, list[int], str, or list[str]
        Plane-type filter applied before candidate selection.

        - ``None``: no filtering.
        - ``int`` or ``list[int]``: keep only terminations with those IDs.
        - ``str`` (element symbol, e.g. ``"O"``): keep terminations
          whose cut plane is **exclusively** that element.  A mixed
          CeO plane would NOT match ``"O"``.
        - ``str`` (plane label, e.g. ``"O4"``): keep terminations whose
          plane label matches.  ``"IrO2"`` matches ``IrO2-a``, ``IrO2-b``
          and ``IrO2-a-recon``; ``"IrO2-a"`` matches ``IrO2-a`` and
          ``IrO2-a-recon``.
        - ``list[str]``: match any entry.  ``["O", "Ce"]`` keeps pure-O
          planes OR pure-Ce planes, but not mixed CeO planes.
    candidates : str
        - ``"best"`` (default): return only the single best candidate
          after plane filtering.  Tasker I/II: fewest bulk bonds broken at
          the cut, then densest surface planes.  Tasker III: lowest dipole,
          then bond score, then distribution score.
        - ``"all"``: return every candidate, generating a plot for each.
    savecandidates : bool
        Save all valid candidates to an extxyz file for visual inspection.

    Returns
    -------
    dict
        Nested dict ``{miller: {plane_id: info}}`` where each ``info``
        dict contains:

        - ``"atoms"`` -- list of ``Atoms`` (one per thickness)
        - ``"tasker_type"`` -- ``"I/II"`` or ``"III"``
        - ``"plane_type"`` -- label of the cut plane (e.g. ``"O4"``,
          ``"O4-recon"``); pass it to ``cutslab(cut_at=...)``
        - ``"plane_counts"`` -- element composition of the cut plane
        - ``"reconstruction"`` -- reconstruction metadata (or None)
        - ``"candidate"`` -- raw scoring dict (Tasker I/II: includes
          ``broken_bonds`` per surface cell, ``broken_bonds_by_pair``
          (e.g. ``{"Ce-O": 8, "Ce-Ce": 12}``) and ``surface_density`` in
          atoms/Å²; IDs are in rank order, ID 0 = best)

        Every slab is checked to be stoichiometric, neutral, non-polar
        and free of internal gaps; a :class:`SlabValidationError` is raised
        otherwise.  A ``ValueError`` explains when no non-polar
        reconstruction exists.
    """
    if candidates not in ("best", "all"):
        raise ValueError(f"candidates must be 'best' or 'all', got {candidates!r}")

    if isinstance(millers, tuple) and len(millers) == 3 and all(isinstance(x, (int, float)) for x in millers):
        millers = [millers]

    result = {}
    for miller in millers:
        result[tuple(miller)] = _generate_for_one_miller(
            bulk_atoms, charges, tuple(miller), layer_thickness_list, bulk_name,
            plane_tol, charge_tol, dipole_tol, vacuum,
            plot, plot_out_dir, verbose,
            bond_threshold, bond_distances,
            prefer_plane, candidates, savecandidates,
        )

    return result


def _generate_for_one_miller(
    bulk_atoms, charges, miller, layer_thickness_list, bulk_name,
    plane_tol, charge_tol, dipole_tol, vacuum,
    plot, plot_out_dir, verbose,
    bond_threshold, bond_distances,
    prefer_plane, candidates, savecandidates,
):
    h, k, l = miller

    if verbose:
        print(f"\nGenerating Tasker slab for {bulk_name} with Miller index ({h}, {k}, {l})\n")

    # Slabs are cut from an index-tagged copy of the bulk so every slab atom
    # can be traced back to its bulk atom (charges, validation).
    charges_list = _charges_to_list(bulk_atoms, charges)
    if len(charges_list) != len(bulk_atoms):
        raise ValueError(
            f"Charges length ({len(charges_list)}) does not match atoms ({len(bulk_atoms)})."
        )
    bulk = _tag_atom_indices(bulk_atoms)

    surf_bulk = build_surface(bulk, miller, layers=1, verbose=verbose)
    atoms_z_matrix, L = compute_projection(
        bulk, surf_bulk, charges, miller, verbose=verbose
    )
    planes = identify_planes(atoms_z_matrix, L, plane_tol=plane_tol, charge_tol=charge_tol)
    reduced_counts = compute_reduced_counts(atoms_z_matrix)
    sequences = enumerate_cut_pairs(planes, L, reduced_counts, charge_tol=charge_tol)
    best_seq = select_best_sequence(sequences, dipole_tol=dipole_tol)

    if best_seq is None:
        raise ValueError("No valid stoichiometry sequences found.")

    planes_sorted = sorted(planes, key=lambda p: p["z_center"] % L)
    plane_names, plane_name_map = assign_plane_names(planes_sorted, atoms=surf_bulk)
    validation = {
        "charges_list": charges_list,
        "reduced_counts": reduced_counts,
        "charge_tol": charge_tol,
        "dipole_tol": dipole_tol,
        "max_gap": _max_z_gap(atoms_z_matrix[:, 1], period=L),
    }

    if verbose:
        valid_sequences = [s for s in sequences if s["is_neutral"] and s["is_stoich"]]
        print("\nValid stoichiometry sequences (charge-neutral, reduced formula):")
        for i, seq in enumerate(valid_sequences):
            tasker_tag = "Tasker II" if seq["dipole_per_fu"] <= dipole_tol else "Tasker III"
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
            bulk, charges, miller, layer_thickness_list, bulk_name,
            planes, planes_sorted, plane_names, plane_name_map,
            sequences, reduced_counts, atoms_z_matrix, L, surf_bulk,
            dipole_tol, vacuum, plane_tol,
            plot, plot_out_dir, verbose,
            bond_threshold, bond_distances,
            prefer_plane, candidates, savecandidates, validation,
        )

    # ---- Tasker III ----
    if verbose:
        print(
            f"No Tasker I/II plane found for miller=({h},{k},{l}) "
            f"on {bulk_name}. Reconstructing Tasker III slab."
        )

    return _tasker3_path(
        bulk, charges, miller, layer_thickness_list, bulk_name,
        planes, planes_sorted, plane_names, plane_name_map,
        reduced_counts, atoms_z_matrix, L, surf_bulk,
        vacuum, plane_tol, charge_tol,
        plot, plot_out_dir, verbose,
        bond_threshold, bond_distances,
        prefer_plane, candidates, savecandidates, validation,
    )


def _tasker12_path(
    bulk_atoms, charges, miller, layer_thickness_list, bulk_name,
    planes, planes_sorted, plane_names, plane_name_map,
    sequences, reduced_counts, atoms_z_matrix, L, surf_bulk,
    dipole_tol, vacuum, plane_tol,
    plot, plot_out_dir, verbose,
    bond_threshold, bond_distances,
    prefer_plane, candidates_mode, savecandidates, validation,
):
    from .plotting import plot_unitcell_atoms
    from .builder import build_cut_slabs
    from .tasker3 import _bonds_across_plane

    h, k, l = miller
    n_pl = len(planes_sorted)

    # One entry per zero-dipole bulk repeat unit (full period); each gives
    # slabs of exactly `lt` repeat units.  Ranked by bulk bonds broken at the
    # cut, then by surface atom density (denser = more compact), then by
    # label, so the order does not depend on the bulk origin.  IDs follow
    # the ranking (ID 0 = best).
    periodic = Atoms(
        numbers=surf_bulk.numbers, positions=surf_bulk.positions,
        cell=surface_bulk_cell(bulk_atoms, miller), pbc=True,
    )
    area = float(np.linalg.norm(np.cross(surf_bulk.cell[0], surf_bulk.cell[1])))
    ranked = []
    for s in sequences:
        if not (s["is_neutral"] and s["is_stoich"] and s["is_full_period"]
                and s["dipole_per_fu"] <= dipole_tol):
            continue
        cut = s["bottom_cut"]
        bot, top = (cut + 1) % n_pl, cut
        z_cut, _ = compute_cut_positions(planes, L, cut, cut)
        seq = dict(s)
        seq["broken_bonds_by_pair"] = _bonds_across_plane(
            periodic, z_cut, L, bond_threshold, bond_distances
        )
        seq["broken_bonds"] = sum(seq["broken_bonds_by_pair"].values())
        seq["surface_density"] = (
            len(planes_sorted[bot]["indices"]) + len(planes_sorted[top]["indices"])
        ) / (2.0 * area)
        key = (seq["broken_bonds"], -round(seq["surface_density"], 6),
               plane_names[bot], plane_names[top], cut)
        ranked.append((key, seq, bot))
    ranked.sort(key=lambda r: r[0])
    zero_dipole = [seq for _, seq, _ in ranked]

    all_terminations = {}
    for tid, (_, seq, bot_plane_idx) in enumerate(ranked):
        all_terminations[tid] = {
            "sequence": seq,
            "plane_type": plane_names[bot_plane_idx],
            "plane_counts": dict(planes_sorted[bot_plane_idx]["counts"]),
        }

    filtered = _filter_by_prefer_plane(all_terminations, prefer_plane)

    if candidates_mode == "best" and filtered:
        # IDs are in rank order: fewest broken bonds, then densest surface.
        best_tid = min(filtered)
        selected = {best_tid: filtered[best_tid]}
    else:
        selected = filtered

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

    output = {}
    for tid, term_info in selected.items():
        seq = term_info["sequence"]
        zbot, ztop = compute_cut_positions(planes, L, seq["bottom_cut"], seq["top_cut"])
        slabs = build_cut_slabs(bulk_atoms, miller, layer_thickness_list, zbot, ztop, L, vacuum)

        if plot:
            n_pl = len(planes_sorted)
            bp = plane_names[(seq["bottom_cut"] + 1) % n_pl]
            tp = plane_names[seq["top_cut"]]
            plot_path = (
                f"{plot_out_dir}/{bulk_name}_hkl_{h}{k}{l}"
                f"_{bp}_{tp}_{tid}.png"
            )
            plot_unitcell_atoms(
                atoms_z_matrix, L, miller,
                out_png=plot_path, plane_tol=plane_tol, planes=planes,
                zbot=zbot, ztop=ztop, dipole=seq["net_dipole"],
                plane_names=plane_names,
            )

        for slab in slabs:
            _finalize_slab(slab, **validation)
            slab.info["bulk_name"] = bulk_name
            slab.info["miller"] = miller

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
    prefer_plane, candidates_mode, savecandidates, validation,
):
    from .plotting import plot_unitcell_atoms
    from .tasker3 import (
        build_adjacency_matrix,
        find_tasker3_candidates,
        build_tasker3_slabs,
        print_adjacency_matrix,
        _midpoint,
        _select_tasker3_candidates,
    )

    h, k, l = miller

    adj = build_adjacency_matrix(
        surf_bulk, bond_threshold=bond_threshold, bond_distances=bond_distances,
        bulk_atoms=bulk_atoms, miller=miller,
    )
    if verbose:
        n_bonds = int(np.sum(adj)) // 2
        print(f"Adjacency: {n_bonds} bonds (threshold {bond_threshold})")
        print_adjacency_matrix(adj, surf_bulk)

    t3_prefer = None
    if isinstance(prefer_plane, str):
        t3_prefer = prefer_plane
    elif isinstance(prefer_plane, (list, tuple)) and prefer_plane and all(isinstance(x, str) for x in prefer_plane):
        t3_prefer = prefer_plane

    t3_candidates = find_tasker3_candidates(
        planes_sorted, atoms_z_matrix, reduced_counts, adj, L,
        surf_bulk=surf_bulk, bond_distances=bond_distances,
        charge_tol=charge_tol, verbose=verbose,
        prefer_plane=t3_prefer,
        plane_names=plane_names,
    )
    t3_candidates = _select_tasker3_candidates(
        t3_candidates, miller, validation["dipole_tol"], charge_tol
    )

    all_terminations = {}
    for tid, cand in enumerate(t3_candidates):
        all_terminations[tid] = {
            "candidate": cand,
            "plane_type": plane_names[cand["cut_plane_idx"]],
            "plane_counts": dict(cand["plane_counts"]),
        }

    filtered = _filter_by_prefer_plane(all_terminations, prefer_plane)

    if candidates_mode == "best" and filtered:
        best_tid = min(filtered, key=lambda t: (
            abs(filtered[t]["candidate"]["net_dipole"]),
            filtered[t]["candidate"]["bond_score"],
            filtered[t]["candidate"]["distribution_score"],
        ))
        selected = {best_tid: filtered[best_tid]}
    else:
        selected = filtered

    if savecandidates:
        from pathlib import Path
        out_p = Path(plot_out_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        all_slabs_for_xyz = []
        for tid, cand in enumerate(t3_candidates):
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
        frac = surf_bulk.get_scaled_positions()
        cut_plane_frac = [
            (int(atoms_z_matrix[idx, 0]), float(frac[idx, 0]) % 1.0, float(frac[idx, 1]) % 1.0)
            for idx in cut_plane["indices"]
        ]

        slabs = build_tasker3_slabs(
            bulk_atoms, miller, layer_thickness_list,
            cut_plane_idx=cand["cut_plane_idx"],
            deletion_mask=cand["deletion_mask"],
            planes_sorted=planes_sorted,
            atoms_z_matrix=atoms_z_matrix,
            L=L, vacuum=vacuum,
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

        if plot:
            z_s = np.array([p["z_center"] % L for p in planes_sorted])
            bot_idx = (cand["cut_plane_idx"] - 1) % len(planes_sorted)
            top_idx = cand["cut_plane_idx"]
            zbot = _midpoint(z_s, L, bot_idx)
            ztop = _midpoint(z_s, L, top_idx)

            recon_names = list(plane_names)
            recon_names[cand["cut_plane_idx"]] = f"{plane_names[cand['cut_plane_idx']]}-recon"

            bp = f"{plane_names[cand['cut_plane_idx']]}-recon"
            tp = bp
            plot_path = (
                f"{plot_out_dir}/{bulk_name}_hkl_{h}{k}{l}"
                f"_{bp}_{tp}_{tid}.png"
            )
            plot_unitcell_atoms(
                atoms_z_matrix, L, miller,
                out_png=plot_path, plane_tol=plane_tol, planes=planes,
                zbot=zbot, ztop=ztop, dipole=cand["net_dipole"],
                plane_names=recon_names,
            )

        for slab in slabs:
            _finalize_slab(slab, **validation)
            slab.info["bulk_name"] = bulk_name
            slab.info["miller"] = miller

        output[tid] = {
            "atoms": slabs,
            "tasker_type": "III",
            "plane_type": f"{cut_plane_name}-recon",
            "plane_counts": dict(cut_plane["counts"]),
            "reconstruction": {
                "cut_plane_name": cut_plane_name,
                "cut_plane_counts": dict(cut_plane["counts"]),
                "cut_plane_frac": cut_plane_frac,
                "delete_info": delete_info,
                "plane_names": plane_names,
                "plane_name_map": plane_name_map,
            },
            "candidate": cand,
        }

    return output
