import numpy as np
from ase.io import read

from .core import (
    _INDEX_KEY,
    _charges_to_list,
    _finalize_slab,
    _find_plane_translation,
    _max_z_gap,
    _planes_from_bulk,
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
    dipole_tol=0.05,
    plot_out_dir=".",
    plot=False,
    verbose=None,
    bond_threshold=(0.85, 1.15),
    bond_distances=None,
    reconstruction=None,
    cut_at="termination",
    cuts="right",
    vacuum=15.0,
    bulk_atoms=None,
    miller=None,
    deform_tol=0.3,
):
    """
    Cut an existing slab into thinner sub-slabs.

    Stoichiometry, charge and dipole of every candidate cut are evaluated
    on the actual (possibly relaxed) atoms of the sub-slab, after any
    reconstruction deletions.

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
        Largest |dipole| per formula unit (e·Å) still treated as zero
        (default 0.05).  Genuinely polar repeat units are ~1-6 e·Å per
        formula unit; relaxed structures may need ~0.3.
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

        - ``"termination"`` (default): cut only at planes with the labels
          of the thick slab's top/bottom planes (a deformed, primed
          surface plane such as ``O4'`` stands for ``O4``).
        - ``"all"``: cut at any plane that gives a stoichiometric,
          charge-neutral, zero-dipole sub-slab.
        - A plane label (e.g. ``"O4"``, or a genslab ``plane_type``) or a
          list of labels: cut only at planes with those labels.
    cuts : str
        ``"right"`` (default) -- fix bottom plane, peel from the top.
        ``"left"`` -- fix top plane, peel from the bottom.
        ``"all"`` -- keep every valid cut.
    vacuum : float
        Vacuum to add (angstrom, per side) to each sub-slab.
    bulk_atoms : Atoms or None
        Bulk unit cell the slab was built from.  When given, every atom is
        assigned to the nearest plane of the bulk (the registry is learned
        from the slab's interior), so relaxed surface planes that rumple or
        shift stay whole, and each plane is labelled by its bulk plane: the
        bulk label (e.g. ``O4``) if it matches within *deform_tol*, a primed
        label (``O4'``) if it is more deformed or has a different
        composition.  Without it, planes come from z-clustering and are
        labelled from the slab alone.
    miller : tuple of int or None
        Miller index of the slab, needed with *bulk_atoms*; defaults to
        ``input_structure.info["miller"]`` (set by genslab).
    deform_tol : float
        RMSD (angstrom, after the best rigid shift) up to which a slab
        plane still counts as its bulk plane (default 0.3).

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
    _put_vacuum_at_boundary(atoms, axis)

    L = float(atoms.cell.lengths()[axis])
    if L <= 0.0:
        raise ValueError("Invalid cell length on selected axis.")

    coords = atoms.positions[:, axis]
    atoms_z_matrix = np.array(
        [[num, z, q] for num, z, q in zip(atoms.numbers, coords, charges_list)]
    )

    vec = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    input_miller = miller if miller is not None else atoms.info.get("miller")
    plot_miller = tuple(input_miller) if input_miller is not None else vec[axis]
    miller_str = "".join(str(i) for i in plot_miller)

    # ---- Planes and their labels ----
    if bulk_atoms is not None:
        if axis != 2:
            raise ValueError("Matching to bulk_atoms needs the surface normal along z (axis=2).")
        if input_miller is None:
            raise ValueError(
                "bulk_atoms needs the slab's Miller index: pass miller=, or use a slab "
                "from generate_slabs_for_miller (it stores slab.info['miller'])."
            )
        planes_sorted, plane_names, reduced_counts = _planes_from_bulk(
            atoms, charges_list, bulk_atoms, tuple(input_miller),
            plane_tol=plane_tol, charge_tol=charge_tol, deform_tol=deform_tol,
        )
    else:
        planes_sorted = sorted(
            identify_planes(atoms_z_matrix, L, plane_tol=plane_tol, charge_tol=charge_tol),
            key=lambda p: p["z_center"] % L,
        )
        plane_names, _ = assign_plane_names(planes_sorted, atoms=atoms, axis=axis)
        reduced_counts = compute_reduced_counts(atoms_z_matrix)
    planes = planes_sorted
    n = len(planes_sorted)
    validation = {
        "charges_list": charges_list,
        "reduced_counts": reduced_counts,
        "charge_tol": charge_tol,
        "dipole_tol": dipole_tol,
        "max_gap": _max_z_gap(coords),
        "axis": axis,
    }

    # Tasker III always requires termination-aware cutting
    if reconstruction is not None and cut_at == "all":
        cut_at = "termination"

    # ---- Reconstruction metadata ----
    # Reconstructed surfaces carry genslab's label (e.g. "O4-recon"): the
    # thick slab's outer planes if they are already reconstructed, and any
    # plane that gets reconstructed when a cut exposes it.
    delete_info = reconstruction["delete_info"] if reconstruction is not None else None
    recon_label = None
    recon_eligible = set()
    if reconstruction is not None:
        cut_plane_counts = reconstruction["cut_plane_counts"]
        recon_label = f"{reconstruction['cut_plane_name']}-recon"
        recon_counts = dict(cut_plane_counts)
        for species, _, _ in delete_info:
            recon_counts[species] = recon_counts.get(species, 0) - 1
        recon_counts = {Z: c for Z, c in recon_counts.items() if c > 0}
        for i in {0, n - 1}:
            if planes_sorted[i]["counts"] == recon_counts:
                plane_names[i] = recon_label
        recon_eligible = {
            i for i, plane in enumerate(planes_sorted)
            if plane["counts"] == cut_plane_counts and i not in (0, n - 1)
        }

    # ---- Planes allowed to become a surface ----
    # A sub-slab is a contiguous run of planes [bottom, top] of the input
    # slab; it never wraps through the vacuum.
    if cut_at == "all":
        valid_boundary_names = set(plane_names)
    elif cut_at == "termination":
        # A deformed (primed) surface plane stands for its bulk plane type.
        ends = {plane_names[0].rstrip("'"), plane_names[-1].rstrip("'")}
        valid_boundary_names = {name for name in plane_names if name.rstrip("'") in ends}
    elif isinstance(cut_at, str):
        matched = {name for name in plane_names if plane_name_matches(cut_at, name)}
        if not matched:
            raise ValueError(
                f"Plane name {cut_at!r} not found. "
                f"Available: {sorted(set(plane_names))}"
            )
        valid_boundary_names = matched
    elif isinstance(cut_at, list):
        valid_boundary_names = set()
        unknown = []
        for query in cut_at:
            matched = {name for name in plane_names if plane_name_matches(query, name)}
            if not matched:
                unknown.append(query)
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
    boundary_indices = sorted(
        {i for i in range(n) if plane_names[i] in valid_boundary_names} | recon_eligible
    )

    # ---- Evaluate every candidate cut on the actual atoms ----
    reference_plane = reconstruction.get("cut_plane_frac") if reconstruction else None
    deletions = {
        p: set(_apply_reconstruction(
            atoms, planes_sorted[p]["indices"], delete_info, axis=axis,
            reference_plane=reference_plane,
        ))
        for p in recon_eligible
    }
    q_all = np.asarray(charges_list, dtype=float)
    species = sorted({int(Z) for Z in atoms.numbers} | {int(Z) for Z in reduced_counts})
    column = {Z: k for k, Z in enumerate(species)}

    def aggregate(indices):
        """[n, sum q, sum z, sum q*z] and element counts of a set of atoms."""
        idx = np.fromiter(indices, dtype=int)
        counts = np.zeros(len(species), dtype=int)
        np.add.at(counts, [column[int(Z)] for Z in atoms.numbers[idx]], 1)
        q, z = q_all[idx], coords[idx]
        return np.array([len(idx), q.sum(), z.sum(), (q * z).sum()]), counts

    plane_sums = [aggregate(p["indices"]) for p in planes_sorted]
    prefix = np.vstack([np.zeros(4), np.cumsum([s for s, _ in plane_sums], axis=0)])
    prefix_counts = np.vstack(
        [np.zeros(len(species), dtype=int), np.cumsum([c for _, c in plane_sums], axis=0)]
    )
    deleted_sums = {p: aggregate(d) for p, d in deletions.items()}

    valid_cuts = []
    for bi in boundary_indices:
        for ti in boundary_indices:
            if ti < bi:
                continue
            sums = prefix[ti + 1] - prefix[bi]
            counts = prefix_counts[ti + 1] - prefix_counts[bi]
            for p in {bi, ti} & set(deleted_sums):
                sums = sums - deleted_sums[p][0]
                counts = counts - deleted_sums[p][1]
            count_map = {Z: int(c) for Z, c in zip(species, counts) if c > 0}
            is_stoich, stoich_k = is_stoichiometric_sequence(count_map, reduced_counts)
            if not is_stoich or set(count_map) - set(reduced_counts):
                continue
            n_atoms, total_q, sum_z, sum_qz = sums
            if abs(total_q) > charge_tol:
                continue
            mu = float(sum_qz - total_q * sum_z / n_atoms)  # about the mean height
            if abs(mu) > dipole_tol * stoich_k:
                continue
            valid_cuts.append({
                "bottom_plane": bi,
                "top_plane": ti,
                "plane_indices": list(range(bi, ti + 1)),
                "n_planes": ti - bi + 1,
                "total_charge": float(total_q),
                "net_dipole": mu,
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
            f"dipole_tol={dipole_tol} per formula unit). Relaxed slabs usually "
            "need dipole_tol~0.3, and rumpled planes a larger plane_tol or "
            "bulk_atoms=." + polar_hint
        )

    # ---- Prepare plot names ----
    highlight_set = set(boundary_indices) | recon_eligible
    plot_names = list(plane_names)
    for i in recon_eligible:
        plot_names[i] = recon_label

    z_s = np.array([p["z_center"] % L for p in planes_sorted])

    # ---- Build sub-slabs ----
    slab_atoms = []
    for cut_idx, cut in enumerate(valid_cuts):
        bi, ti = cut["bottom_plane"], cut["top_plane"]
        drop = set().union(*(deletions.get(p, set()) for p in {bi, ti}))
        atom_indices = sorted(
            {i for pidx in cut["plane_indices"] for i in planes_sorted[pidx]["indices"]} - drop
        )
        slab = atoms[atom_indices]
        apply_vacuum_to_slab(slab, vacuum=vacuum, axis=axis)

        bp = plot_names[bi]
        tp = plot_names[ti]
        slab.info["cut_bottom_plane"] = bp
        slab.info["cut_top_plane"] = tp
        slab.info["cut_bottom_idx"] = bi
        slab.info["cut_top_idx"] = ti
        slab.info["cut_n_planes"] = cut["n_planes"]
        _finalize_slab(slab, **validation)

        if plot:
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
                atoms_z_matrix, L, plot_miller,
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


def _put_vacuum_at_boundary(atoms, axis):
    """
    Make a slab contiguous along *axis*, with its largest gap (the vacuum) at
    the cell boundary, e.g. for slabs centred at the origin and wrapped.

    Atoms move by whole lattice vectors plus one rigid shift of the whole
    structure, so the slab itself is unchanged.
    """
    if len(atoms) < 2:
        return
    frac = atoms.get_scaled_positions(wrap=False)
    f = frac[:, axis] % 1.0
    ordered = np.sort(f)
    gaps = np.diff(np.append(ordered, ordered[0] + 1.0))
    start = ordered[(int(np.argmax(gaps)) + 1) % len(ordered)]
    frac[:, axis] = (f - start) % 1.0
    atoms.set_scaled_positions(frac)


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
        t = _find_plane_translation(reference_plane, target, cell2d, tol)
        if t is not None:
            shift = t
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
