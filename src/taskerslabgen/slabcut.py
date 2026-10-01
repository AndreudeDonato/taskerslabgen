import numpy as np
from ase.io import read

from .core import (
    _INDEX_KEY,
    _charges_to_list,
    _finalize_slab,
    _find_plane_alignment,
    _max_z_gap,
    identify_planes,
    compute_reduced_counts,
    apply_vacuum_to_slab,
    assign_plane_names,
    is_stoichiometric_sequence,
    plane_name_matches,
)


def cutslab(
    input_structure,
    charges,
    axis=2,
    plane_tol=None,
    charge_tol=1e-3,
    dipole_tol=1e-6,
    plot_out_dir=".",
    plot=False,
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
    plane_tol : float or None
        Largest z-gap (angstrom) between neighbouring atoms of one plane
        (single-linkage clustering).  ``None`` (default) uses 0.1 Å.
    charge_tol : float
        Tolerance for charge neutrality.
    dipole_tol : float
        Threshold below which the dipole is considered zero.  Relaxed slabs
        usually need a larger value.
    plot_out_dir : str
        Directory for output plots.
    plot : bool
        Generate stacking-axis plots for each sub-slab (default ``False``).
    verbose : bool or None
        Print detailed cut information.
    bond_threshold : tuple of float
        Unused; kept for backward compatibility (the Tasker III fallback
        was removed: it treated the slab as a periodic bulk cell).
    bond_distances : dict or None
        Unused; kept for backward compatibility.
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
        - ``"all"``: cut at any plane that gives a stoichiometric,
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
        Sub-slabs sorted from smallest to largest by atom count.  Each one
        is checked to be stoichiometric, neutral, non-polar and free of
        internal gaps (:class:`SlabValidationError` otherwise).
        Each ``Atoms`` object has metadata in ``.info``:
        ``cut_bottom_plane``, ``cut_top_plane``, ``cut_bottom_idx``,
        ``cut_top_idx``, ``cut_n_planes``.
    """
    from .plotting import plot_unitcell_atoms

    if cuts not in ("right", "left", "all"):
        raise ValueError(
            f"Unknown cuts mode: {cuts!r}. Must be 'right', 'left', or 'all'."
        )

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
    # Track input indices through slicing (charges for validation).
    atoms.arrays[_INDEX_KEY] = np.arange(len(atoms))

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
    validation = {
        "charges_list": charges_list,
        "reduced_counts": reduced_counts,
        "charge_tol": charge_tol,
        "dipole_tol": dipole_tol,
        "max_gap": _max_z_gap(coords),
        "axis": axis,
    }

    vec = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    miller = vec[axis]
    input_miller = atoms.info.get("miller")
    if input_miller is not None:
        miller_str = "".join(str(i) for i in input_miller)
    else:
        miller_str = "".join(str(i) for i in miller)

    plane_names, _ = assign_plane_names(planes_sorted, atoms=atoms, axis=axis)

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

    # ---- Planes allowed to become a surface ----
    # A sub-slab is a contiguous run of planes [bottom, top] of the input
    # slab; it never wraps through the vacuum.
    if cut_at == "all":
        valid_boundary_names = set(plane_names)
    elif cut_at == "termination":
        valid_boundary_names = {plane_names[0], plane_names[-1]}
    elif isinstance(cut_at, str):
        matched = {n for n in plane_names if plane_name_matches(cut_at, n)}
        if not matched:
            raise ValueError(
                f"Plane name {cut_at!r} not found. "
                f"Available: {sorted(set(plane_names))}"
            )
        valid_boundary_names = matched
    elif isinstance(cut_at, list):
        valid_boundary_names = set()
        unknown = []
        for q in cut_at:
            matched = {n for n in plane_names if plane_name_matches(q, n)}
            if not matched:
                unknown.append(q)
            else:
                valid_boundary_names |= matched
        if unknown:
            raise ValueError(
                f"Unknown plane names: {unknown}. "
                f"Available: {sorted(set(plane_names))}"
            )
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

    if valid_cuts and cuts == "right":
        fixed_bot = min(c["bottom_plane"] for c in valid_cuts)
        valid_cuts = [
            c for c in valid_cuts if c["bottom_plane"] == fixed_bot
        ]
    elif valid_cuts and cuts == "left":
        fixed_top = max(c["top_plane"] for c in valid_cuts)
        valid_cuts = [
            c for c in valid_cuts if c["top_plane"] == fixed_top
        ]

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
        polar_hint = "" if reconstruction is not None else (
            " If the slab is polar (Tasker III), build it with "
            "generate_slabs_for_miller and pass reconstruction=term['reconstruction']."
        )
        raise ValueError(
            "No stoichiometric, charge-neutral, zero-dipole cuts found "
            f"matching cut_at={cut_at!r} (charge_tol={charge_tol}, "
            f"dipole_tol={dipole_tol}). Relaxed slabs usually need a larger "
            "dipole_tol, and rumpled planes a larger plane_tol." + polar_hint
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
    reference_plane = reconstruction.get("cut_plane_frac") if reconstruction else None
    slab_atoms = []
    for cut_idx, cut in enumerate(valid_cuts):
        atom_indices = []
        for pidx in cut["plane_indices"]:
            atom_indices.extend(planes_sorted[pidx]["indices"])
        atom_indices = sorted(set(atom_indices))
        slab = atoms[atom_indices]

        if reconstruction and delete_info:
            position = {orig: k for k, orig in enumerate(atom_indices)}
            to_delete = set()
            surface_pidxs = (
                [cut["bottom_plane"]]
                if cut["bottom_plane"] == cut["top_plane"]
                else [cut["bottom_plane"], cut["top_plane"]]
            )
            for pidx in surface_pidxs:
                if pidx not in recon_eligible:
                    continue
                surface_indices = [position[j] for j in planes_sorted[pidx]["indices"]]
                to_delete.update(_apply_reconstruction(
                    slab, surface_indices, delete_info, axis=axis,
                    reference_plane=reference_plane,
                ))
            if to_delete:
                slab = slab[[i for i in range(len(slab)) if i not in to_delete]]

        apply_vacuum_to_slab(slab, vacuum=vacuum, axis=axis)

        bp = plane_names[cut["bottom_plane"]]
        tp = plane_names[cut["top_plane"]]
        slab.info["cut_bottom_plane"] = bp
        slab.info["cut_top_plane"] = tp
        slab.info["cut_bottom_idx"] = cut["bottom_plane"]
        slab.info["cut_top_idx"] = cut["top_plane"]
        slab.info["cut_n_planes"] = cut["n_planes"]
        _finalize_slab(slab, **validation)

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

    slab_atoms.sort(key=len)
    return slab_atoms


# ---- Private helpers -------------------------------------------------------


def _apply_reconstruction(slab, plane_indices, delete_info, axis=2,
                          reference_plane=None, tol=0.5):
    """
    Pick the atoms of a surface plane to delete for a Tasker III pattern.

    *delete_info* lists ``(Z, fx, fy)`` of the deleted atoms in the
    reference plane.  With *reference_plane* (all atoms of that plane as
    ``(Z, fx, fy)``) the reference is first aligned onto the target plane
    by an in-plane translation, so the same pattern is reproduced wherever
    the plane copy sits; otherwise the atoms nearest to the stored
    positions are taken.

    Returns list of atom indices (in *slab*) to delete.
    """
    frac = slab.get_scaled_positions()
    ab_axes = [i for i in range(3) if i != axis]
    shift = np.zeros(2)
    if reference_plane is not None:
        target = [
            (int(slab.numbers[j]), frac[j, ab_axes[0]], frac[j, ab_axes[1]])
            for j in plane_indices
        ]
        cell2d = np.array(slab.cell)[np.ix_(ab_axes, ab_axes)]
        match = _find_plane_alignment(reference_plane, target, cell2d, tol)
        if match is not None:
            shift = match[1]
    to_delete = set()
    for species, fx, fy in delete_info:
        fx, fy = fx + shift[0], fy + shift[1]
        best_j = None
        best_d = np.inf
        for j in plane_indices:
            if j in to_delete:
                continue
            if slab.numbers[j] != species:
                continue
            dfx = abs((frac[j, ab_axes[0]] - fx) % 1.0)
            dfy = abs((frac[j, ab_axes[1]] - fy) % 1.0)
            dfx = min(dfx, 1.0 - dfx)
            dfy = min(dfy, 1.0 - dfy)
            d = np.sqrt(dfx**2 + dfy**2)
            if d < best_d:
                best_d = d
                best_j = j
        if best_j is not None:
            to_delete.add(best_j)
    return sorted(to_delete)
