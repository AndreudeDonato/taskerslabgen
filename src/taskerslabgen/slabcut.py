import numpy as np
from ase.io import read

from .core import (
    _charges_to_list,
    identify_planes,
    compute_reduced_counts,
    enumerate_cut_pairs,
    compute_cut_positions,
    apply_vacuum_to_slab,
    assign_plane_names,
    is_stoichiometric_sequence,
)


def cutslab(
    input_structure,
    charges,
    axis=2,
    plane_tol=0.05,
    charge_tol=1e-3,
    dipole_tol=1e-6,
    plot_out_dir=".",
    plot=True,
    verbose=None,
    bond_threshold=(0.85, 1.15),
    bond_distances=None,
    reconstruction=None,
    cut_at="termination",
    cuts="right",
    vacuum=15.0,
):
    """
    Cut an existing slab into thinner sub-slabs.

    Parameters
    ----------
    input_structure : Atoms or str/Path
        The thick slab to cut.  An ASE ``Atoms`` object or a file path
        readable by ``ase.io.read``.
    charges : dict or list
        Formal charges (same format as :func:`compute_projection`).
    axis : int
        Cartesian axis perpendicular to the surface (0, 1, or 2).
    plane_tol : float
        Tolerance (angstrom) for grouping atoms into planes.
    charge_tol : float
        Tolerance for charge neutrality.
    dipole_tol : float
        Threshold below which the dipole is considered zero.
    plot_out_dir : str
        Directory for output plots.
    plot : bool
        Generate stacking-axis plots for each sub-slab.
    verbose : bool or None
        Print detailed cut information.
    bond_threshold : tuple of float
        ``(lo, hi)`` scaling factors for the adjacency matrix (only
        used in Tasker III fallback path).
    bond_distances : dict or None
        Per-pair bond reference distances.
    reconstruction : dict or None
        Tasker III reconstruction pattern (the ``term["reconstruction"]``
        dict from :func:`generate_slabs_for_miller`).  When provided,
        newly exposed planes matching the reconstruction pattern get the
        same atomic deletion applied.  Forces ``cut_at="termination"``
        if ``cut_at`` was ``"all"``.
    cut_at : str or list[str]
        Controls where cuts are placed:

        - ``"termination"`` (default): cut only at planes matching the
          thick slab's top/bottom plane types.
        - ``"all"``: cut at any boundary that gives a stoichiometric,
          charge-neutral, zero-dipole sub-slab.
        - A plane name (e.g. ``"P0"``) or list of names: cut only at
          boundaries where those plane types are exposed.
    cuts : str
        ``"right"`` (default) -- fix bottom plane, peel from the top.
        ``"left"`` -- fix top plane, peel from the bottom.
        ``"all"`` -- keep every valid cut.
    vacuum : float
        Vacuum to add (angstrom, per side) to each sub-slab.

    Returns
    -------
    list of Atoms
        Sub-slabs sorted from smallest to largest by atom count.
        Each ``Atoms`` object has metadata in ``.info``:
        ``cut_bottom_plane``, ``cut_top_plane``, ``cut_bottom_idx``,
        ``cut_top_idx``, ``cut_n_planes``.
    """
    from .plotting import plot_unitcell_atoms

    # ---- Parse input ----
    if hasattr(input_structure, "positions"):
        atoms = input_structure.copy()
        stem = input_structure.info.get("bulk_name", "structure")
    else:
        atoms = read(str(input_structure))
        stem = (
            getattr(input_structure, "stem", None)
            or str(input_structure).split("/")[-1].split(".")[0]
        )

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

    planes = identify_planes(
        atoms_z_matrix, L, plane_tol=plane_tol, charge_tol=charge_tol
    )
    reduced_counts = compute_reduced_counts(atoms_z_matrix)
    planes_sorted = sorted(planes, key=lambda p: p["z_center"] % L)
    n = len(planes_sorted)

    vec = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    miller = vec[axis]
    input_miller = atoms.info.get("miller")
    if input_miller is not None:
        miller_str = "".join(str(i) for i in input_miller)
    else:
        miller_str = "".join(str(i) for i in miller)

    plane_names, plane_name_map = assign_plane_names(
        planes_sorted, atoms=atoms, axis=axis
    )

    # Tasker III always requires termination-aware cutting
    if reconstruction is not None and cut_at == "all":
        cut_at = "termination"

    # ---- Parse reconstruction metadata ----
    cut_plane_counts = None
    delete_info = None
    recon_del_counts = {}
    recon_del_charge = 0.0
    if reconstruction is not None:
        cut_plane_counts = reconstruction["cut_plane_counts"]
        delete_info = reconstruction["delete_info"]
        charge_map = {}
        for Z_val in set(int(atoms.numbers[j]) for j in range(len(atoms))):
            for j in range(len(atoms)):
                if int(atoms.numbers[j]) == Z_val:
                    charge_map[Z_val] = charges_list[j]
                    break
        for species, _, _ in delete_info:
            recon_del_counts[species] = recon_del_counts.get(species, 0) + 1
            recon_del_charge += charge_map.get(species, 0.0)

    # ================================================================
    # PATH C  –  cut_at="all": enumerate all zero-dipole cuts
    # ================================================================
    if cut_at == "all":
        sequences = enumerate_cut_pairs(
            planes, L, reduced_counts, charge_tol=charge_tol
        )
        zero_dipole = [
            s for s in sequences
            if s["is_stoich"]
            and s["is_neutral"]
            and abs(s["net_dipole"]) <= dipole_tol
        ]
        valid_sequences = _filter_sequences_by_cuts(zero_dipole, cuts)
        valid_sequences.sort(key=lambda s: (s["bottom_cut"], s["top_cut"]))

        if valid_sequences:
            if plot:
                plot_path = (
                    f"{plot_out_dir}/{stem}_hkl_{miller_str}_planes.png"
                )
                plot_unitcell_atoms(
                    atoms_z_matrix, L, miller,
                    out_png=plot_path, plane_tol=plane_tol, planes=planes,
                    zbot=None, ztop=None, dipole=0,
                    plane_names=list(plane_names),
                )

            slab_atoms = _build_slabs_from_sequences(
                atoms, planes_sorted, valid_sequences, L,
                reconstruction, cut_plane_counts, delete_info,
                {0, n - 1}, plane_tol, vacuum, axis, verbose,
                plane_names=plane_names,
            )
            slab_atoms.sort(key=len)
            return slab_atoms

        # Tasker III fallback (independent discovery, only without
        # user-provided reconstruction)
        if reconstruction is None:
            return _tasker3_fallback(
                atoms, atoms_z_matrix, planes, planes_sorted,
                reduced_counts, plane_names, plane_name_map,
                L, axis, miller, stem, charges,
                plane_tol, charge_tol, dipole_tol,
                bond_threshold, bond_distances,
                plot_out_dir, plot, verbose, vacuum,
                plot_unitcell_atoms,
                miller_str=miller_str,
            )

        raise ValueError(
            "No valid cuts found in cut_at='all' mode "
            "(even after accounting for reconstruction)."
        )

    # ================================================================
    # PATH A  –  cut_at="termination" or specific plane name(s)
    # ================================================================
    if cut_at == "termination":
        valid_boundary_names = {plane_names[0], plane_names[-1]}
    elif isinstance(cut_at, str):
        if cut_at not in set(plane_names):
            raise ValueError(
                f"Plane name {cut_at!r} not found. "
                f"Available: {sorted(set(plane_names))}"
            )
        valid_boundary_names = {cut_at}
    elif isinstance(cut_at, list):
        unknown = set(cut_at) - set(plane_names)
        if unknown:
            raise ValueError(
                f"Unknown plane names: {unknown}. "
                f"Available: {sorted(set(plane_names))}"
            )
        valid_boundary_names = set(cut_at)
    else:
        raise ValueError(
            f"Invalid cut_at={cut_at!r}. Must be 'all', 'termination', "
            f"a plane name, or list of plane names."
        )

    boundary_indices = [
        i for i in range(n) if plane_names[i] in valid_boundary_names
    ]

    endpoint_indices = {0, n - 1}

    recon_eligible = set()
    if reconstruction and cut_plane_counts:
        for i, plane in enumerate(planes_sorted):
            if plane["counts"] == cut_plane_counts and i not in endpoint_indices:
                recon_eligible.add(i)
        boundary_indices = sorted(set(boundary_indices) | recon_eligible)

    z_arr = np.array([p["z_center"] for p in planes_sorted])
    q_arr = np.array([p["q_total"] for p in planes_sorted])

    valid_cuts = []
    for bi in boundary_indices:
        for ti in boundary_indices:
            if ti < bi:
                continue

            seq_indices = list(range(bi, ti + 1))
            seq_counts = {}
            for pi in seq_indices:
                for Z, c in planes_sorted[pi]["counts"].items():
                    seq_counts[Z] = seq_counts.get(Z, 0) + c

            adj_counts = dict(seq_counts)
            adj_q_offset = 0.0
            n_recon_surfaces = 0
            if bi == ti:
                if bi in recon_eligible:
                    n_recon_surfaces = 1
            else:
                if bi in recon_eligible:
                    n_recon_surfaces += 1
                if ti in recon_eligible:
                    n_recon_surfaces += 1
            for Z, nd in recon_del_counts.items():
                adj_counts[Z] = adj_counts.get(Z, 0) - nd * n_recon_surfaces
            adj_q_offset = recon_del_charge * n_recon_surfaces

            is_stoich, stoich_k = is_stoichiometric_sequence(
                adj_counts, reduced_counts
            )
            if not is_stoich:
                continue

            total_q = float(np.sum(q_arr[seq_indices])) - adj_q_offset
            if abs(total_q) > charge_tol:
                continue

            z_seq = z_arr[seq_indices]
            z_center = 0.5 * (z_seq[0] + z_seq[-1])
            q_adj = np.array(q_arr[seq_indices], dtype=float)
            if bi == ti:
                if bi in recon_eligible:
                    q_adj[0] -= recon_del_charge
            else:
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

    if cuts == "right" and valid_cuts:
        fixed_bot = min(c["bottom_plane"] for c in valid_cuts)
        valid_cuts = [
            c for c in valid_cuts if c["bottom_plane"] == fixed_bot
        ]
    elif cuts == "left" and valid_cuts:
        fixed_top = max(c["top_plane"] for c in valid_cuts)
        valid_cuts = [
            c for c in valid_cuts if c["top_plane"] == fixed_top
        ]
    elif cuts == "all":
        pass
    else:
        raise ValueError(
            f"Unknown cuts mode: {cuts!r}. "
            f"Must be 'right', 'left', or 'all'."
        )

    valid_cuts.sort(key=lambda c: c["n_planes"])

    if verbose:
        print(f"\nPlane stacking: {' '.join(plane_names)}")
        print(f"Valid boundary types: {sorted(valid_boundary_names)}")
        if recon_eligible:
            print(f"Reconstruction-eligible planes: {sorted(recon_eligible)}")
        print(f"\nValid cuts (mode={cuts!r}): {len(valid_cuts)}")
        for i, cut in enumerate(valid_cuts):
            bn = plane_names[cut["bottom_plane"]]
            tn = plane_names[cut["top_plane"]]
            print(
                f"  {i:3d}  {bn}[{cut['bottom_plane']}]"
                f"-{tn}[{cut['top_plane']}]  "
                f"({cut['n_planes']} planes)  "
                f"Q={cut['total_charge']:+.3f}  "
                f"mu={cut['net_dipole']:+.4e}"
            )

    if not valid_cuts:
        raise ValueError(
            "No stoichiometric, charge-neutral, zero-dipole cuts found "
            f"matching cut_at={cut_at!r}."
        )

    # ---- Prepare plot names ----
    highlight_set = set(boundary_indices) | recon_eligible
    plot_names = list(plane_names)
    cut_plane_name = (
        reconstruction.get("cut_plane_name") if reconstruction else None
    )
    for i in range(n):
        if i in recon_eligible and cut_plane_name:
            plot_names[i] = f"{plane_names[i]}-recon"

    z_s = np.array([p["z_center"] % L for p in planes_sorted])

    # ---- Build sub-slabs with optional reconstruction ----
    slab_atoms = []
    for cut_idx, cut in enumerate(valid_cuts):
        atom_indices = []
        for pidx in cut["plane_indices"]:
            atom_indices.extend(planes_sorted[pidx]["indices"])
        atom_indices = sorted(set(atom_indices))
        slab = atoms[atom_indices]

        if reconstruction and delete_info:
            all_delete = []
            surface_pidxs = (
                [cut["bottom_plane"]]
                if cut["bottom_plane"] == cut["top_plane"]
                else [cut["bottom_plane"], cut["top_plane"]]
            )
            for pidx in surface_pidxs:
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
                keep = [
                    i for i in range(len(slab)) if i not in set(all_delete)
                ]
                slab = slab[keep]

        apply_vacuum_to_slab(slab, vacuum=vacuum, axis=axis)

        bp = plane_names[cut["bottom_plane"]]
        tp = plane_names[cut["top_plane"]]
        slab.info["cut_bottom_plane"] = bp
        slab.info["cut_top_plane"] = tp
        slab.info["cut_bottom_idx"] = cut["bottom_plane"]
        slab.info["cut_top_idx"] = cut["top_plane"]
        slab.info["cut_n_planes"] = cut["n_planes"]

        if plot:
            bi = cut["bottom_plane"]
            ti = cut["top_plane"]
            zbot_mid = 0.5 * (
                z_s[(bi - 1) % n] + z_s[bi]
            ) if bi > 0 else z_s[bi] * 0.5
            ztop_mid = 0.5 * (
                z_s[ti] + z_s[(ti + 1) % n]
            ) if ti < n - 1 else 0.5 * (z_s[ti] + L)
            plot_path = (
                f"{plot_out_dir}/{stem}_hkl_{miller_str}"
                f"_cut_{cut_idx}_{bp}_{tp}.png"
            )
            plot_unitcell_atoms(
                atoms_z_matrix, L, miller,
                out_png=plot_path, plane_tol=plane_tol, planes=planes,
                zbot=zbot_mid, ztop=ztop_mid, dipole=cut["net_dipole"],
                matched_planes=highlight_set,
                plane_names=plot_names,
                title=(
                    f"cutslab {stem} hkl={miller_str} "
                    f"cut {cut_idx} ({bp}-{tp}, "
                    f"{cut['n_planes']} planes)"
                ),
            )

        slab_atoms.append(slab)

    return slab_atoms


# ---- Private helpers -------------------------------------------------------


def _filter_sequences_by_cuts(zero_dipole, cuts):
    """Apply the ``cuts`` mode filter to a list of zero-dipole sequences."""
    if cuts == "all":
        return list(zero_dipole)

    if not zero_dipole:
        return []

    all_bot = sorted(set(s["bottom_cut"] for s in zero_dipole))
    all_top = sorted(set(s["top_cut"] for s in zero_dipole), reverse=True)

    if cuts == "right":
        fixed_bot = all_bot[0]
        return [s for s in zero_dipole if s["bottom_cut"] == fixed_bot]
    elif cuts == "left":
        fixed_top = all_top[0]
        return [s for s in zero_dipole if s["top_cut"] == fixed_top]
    elif cuts == "all":
        return list(zero_dipole)
    else:
        raise ValueError(
            f"Unknown cuts mode: {cuts!r}. "
            f"Must be 'right', 'left', or 'all'."
        )


def _build_slabs_from_sequences(
    atoms, planes_sorted, valid_sequences, L,
    reconstruction, cut_plane_counts, delete_info,
    endpoint_indices, plane_tol, vacuum, axis, verbose,
    plane_names=None,
):
    """Build Atoms sub-slabs from validated sequences (``cut_at='all'`` path)."""
    n = len(planes_sorted)

    recon_eligible = set()
    if reconstruction and cut_plane_counts:
        for i, plane in enumerate(planes_sorted):
            if plane["counts"] == cut_plane_counts and i not in endpoint_indices:
                recon_eligible.add(i)

    slab_atoms = []
    for idx, seq in enumerate(valid_sequences, start=1):
        zbot, ztop = compute_cut_positions(
            sorted(
                [p for p in [planes_sorted[i] for i in range(n)]],
                key=lambda p: p["z_center"] % L,
            ),
            L, seq["bottom_cut"], seq["top_cut"],
        )
        atom_indices = []
        for pidx in seq["plane_indices"]:
            atom_indices.extend(planes_sorted[pidx]["indices"])
        atom_indices = sorted(set(atom_indices))
        slab = atoms[atom_indices]

        if reconstruction and delete_info and recon_eligible:
            pi_bot = seq["plane_indices"][0]
            pi_top = seq["plane_indices"][-1]
            all_delete = []
            for pidx in [pi_bot, pi_top]:
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
                keep = [
                    i for i in range(len(slab)) if i not in set(all_delete)
                ]
                slab = slab[keep]

        apply_vacuum_to_slab(slab, vacuum=vacuum, axis=axis)

        pi_bot_name = seq["plane_indices"][0]
        pi_top_name = seq["plane_indices"][-1]
        slab.info["cut_bottom_plane"] = (
            plane_names[pi_bot_name] if plane_names else f"P{pi_bot_name}"
        )
        slab.info["cut_top_plane"] = (
            plane_names[pi_top_name] if plane_names else f"P{pi_top_name}"
        )
        slab.info["cut_bottom_idx"] = pi_bot_name
        slab.info["cut_top_idx"] = pi_top_name
        slab.info["cut_n_planes"] = len(seq["plane_indices"])

        slab_atoms.append(slab)

        if verbose:
            bottom_edge = (
                f"{seq['bottom_cut']}"
                f"-{(seq['bottom_cut'] + 1) % n}"
            )
            top_edge = (
                f"{seq['top_cut']}"
                f"-{(seq['top_cut'] + 1) % n}"
            )
            print(
                f"{idx:3d}  bottom_cut={bottom_edge} top_cut={top_edge}  "
                f"Q={seq['total_charge']:+.3f}  "
                f"mu={seq['net_dipole']:+.4e}  "
                f"z_center={seq['z_center']:.3f}"
            )

    return slab_atoms


def _tasker3_fallback(
    atoms, atoms_z_matrix, planes, planes_sorted,
    reduced_counts, plane_names, plane_name_map,
    L, axis, miller, stem, charges,
    plane_tol, charge_tol, dipole_tol,
    bond_threshold, bond_distances,
    plot_out_dir, plot, verbose, vacuum,
    plot_unitcell_atoms, miller_str="",
):
    """Tasker III independent discovery fallback for ``cut_at='all'``."""
    print(
        f"No Tasker I/II cut found for axis={axis}. "
        f"Reconstructing Tasker III slab."
    )

    from .tasker3 import (
        build_adjacency_matrix,
        find_tasker3_candidates,
        print_adjacency_matrix,
        _midpoint,
    )

    adj = build_adjacency_matrix(
        atoms, bond_threshold=bond_threshold, bond_distances=bond_distances,
    )
    if verbose:
        n_bonds = int(np.sum(adj)) // 2
        print(f"Adjacency: {n_bonds} bonds (threshold {bond_threshold})")
        print_adjacency_matrix(adj, atoms)

    candidates = find_tasker3_candidates(
        planes_sorted, atoms_z_matrix, reduced_counts, adj, L,
        surf_bulk=atoms, bond_distances=bond_distances,
        charge_tol=charge_tol, verbose=verbose,
        plane_names=plane_names,
    )
    if not candidates:
        raise ValueError("No Tasker III reconstruction candidates found.")

    best = candidates[0]
    if verbose:
        print(
            f"\n-> Best Tasker III: plane {best['cut_plane_idx']}  "
            f"mu={best['net_dipole']:+.4e}  "
            f"bonds_broken={best['bond_score']}  "
            f"distr={best['distribution_score']:+.4f}\n"
        )

    if plot:
        z_s = np.array([p["z_center"] % L for p in planes_sorted])
        bot_idx = (best["cut_plane_idx"] - 1) % len(planes_sorted)
        top_idx = best["cut_plane_idx"]
        zbot = _midpoint(z_s, L, bot_idx)
        ztop = _midpoint(z_s, L, top_idx)
        plot_path = f"{plot_out_dir}/{stem}_hkl_{miller_str}_tasker3.png"
        plot_unitcell_atoms(
            atoms_z_matrix, L, miller,
            out_png=plot_path, plane_tol=plane_tol, planes=planes,
            zbot=zbot, ztop=ztop, dipole=best["net_dipole"],
        )

    deleted_set = set(best["deletion_mask"])
    keep = [i for i in range(len(atoms)) if i not in deleted_set]
    slab = atoms[keep]
    apply_vacuum_to_slab(slab, vacuum=vacuum, axis=axis)

    cut_pi = best["cut_plane_idx"]
    slab.info["cut_bottom_plane"] = plane_names[0]
    slab.info["cut_top_plane"] = plane_names[-1]
    slab.info["cut_bottom_idx"] = 0
    slab.info["cut_top_idx"] = len(planes_sorted) - 1
    slab.info["cut_n_planes"] = len(planes_sorted)

    return [slab]


def _apply_reconstruction(slab, plane_indices, delete_info, axis=2):
    """
    Apply a Tasker III reconstruction pattern to a set of atoms
    identified as a surface plane.

    Returns list of atom indices (in *slab*) to delete.
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
