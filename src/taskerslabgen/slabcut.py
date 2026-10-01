import warnings
from collections import Counter

import numpy as np
from ase.io import read
from scipy.optimize import linear_sum_assignment

from .core import (
    _INDEX_KEY,
    _charges_to_list,
    _finalize_slab,
    _frac_distance,
    _make_plane,
    _plane_translations,
    _planes_from_bulk,
    _shift_matches,
    _supercell_matrix,
    _tile_plane,
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
    charges : dict, list or None
        Formal charges: a dict by element, one value per atom, or ``None``
        to use the charges stored on the slab (calculator results
        ``"charges"``, else ``initial_charges``), e.g. computed charges of a
        relaxed slab.
    axis : int
        Cartesian axis perpendicular to the surface (0, 1, or 2).
    plane_tol : float or None
        Largest z-gap (angstrom) between neighbouring atoms of one plane
        (single-linkage clustering).  ``None`` (default) uses 0.1 Å.
    charge_tol : float
        Largest |net charge| per formula unit (e) treated as neutral
        (default 1e-3).
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
        dict from :func:`generate_slabs_for_miller`, also after a JSON round
        trip).  Every interior copy of the reconstructed bulk plane gets the
        same deletions when a cut exposes it, placed as genslab places them
        (aligned on the plane and its neighbours, a whole number of repeat
        units from the slab's reconstructed surface).  Works for in-plane
        supercells of the slab genslab made.  Raises if no plane of the slab
        matches.  Forces ``cut_at="termination"`` if ``cut_at`` was
        ``"all"``; with an explicit ``cut_at``, copies are exposed only if
        the reconstructed label (e.g. ``"O4-recon"``) is selected.
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
        Bulk the slab was built from: its unit cell or a supercell of it
        (e.g. a relaxed bulk calculation), up to a few per cent of strain.
        When given, every atom is assigned to the nearest plane of the bulk
        (the registry and the period along the normal are learned from the
        slab's interior, from single atoms if relaxation split every plane),
        so relaxed surface planes that rumple or shift stay whole, and each
        plane is labelled by its bulk plane: the bulk label (e.g. ``O4``) if
        it matches within *deform_tol*, a primed label (``O4'``) if it is
        more deformed or has a different composition.  Without it, planes
        come from z-clustering and are labelled from the slab alone.
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
        is checked to be stoichiometric, neutral and non-polar
        (:class:`SlabValidationError` otherwise).
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
        # From the original: Atoms.copy() drops the calculator and its charges.
        charges_list = _charges_to_list(input_structure, charges)
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

    recon = None
    if reconstruction is not None:
        ab_axes = [i for i in range(3) if i != axis]
        cell2d = np.array(atoms.cell)[np.ix_(ab_axes, ab_axes)]
        recon = _prepare_reconstruction(reconstruction, cell2d)

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
        if recon is not None:
            planes_sorted, plane_names = _merge_surface_planes(
                planes_sorted, plane_names, recon["recon_counts"], atoms_z_matrix, charge_tol
            )
    else:
        planes_sorted = sorted(
            identify_planes(atoms_z_matrix, L, plane_tol=plane_tol, charge_tol=charge_tol),
            key=lambda p: p["z_center"] % L,
        )
        if recon is not None:
            planes_sorted, _ = _merge_surface_planes(
                planes_sorted, None, recon["recon_counts"], atoms_z_matrix, charge_tol
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
        "axis": axis,
    }

    # Tasker III always requires termination-aware cutting
    if reconstruction is not None and cut_at == "all":
        cut_at = "termination"

    # ---- Reconstruction metadata ----
    # Reconstructed surfaces carry genslab's label (e.g. "O4-recon"): the
    # thick slab's outer planes if they are already reconstructed, and every
    # interior copy of the cut plane, which gets the pattern when a cut
    # exposes it.
    recon_label = None
    deletions = {}
    if recon is not None:
        recon_label = recon["recon_label"]
        outer = [i for i in dict.fromkeys((0, n - 1))
                 if planes_sorted[i]["counts"] == recon["recon_counts"]]
        for i in outer:
            plane_names[i] = recon_label
        deletions = _reconstruction_deletions(atoms, planes_sorted, outer, recon, cell2d, ab_axes)
        if not deletions and not outer:
            raise ValueError(
                f"The reconstruction ({recon_label}) matches no plane of the slab: no "
                "plane has the composition and geometry of the reconstructed plane or "
                "of its bulk plane.  Check that the slab and the reconstruction come "
                "from the same bulk and Miller index."
            )
    recon_eligible = set(deletions)

    label_hint = "" if bulk_atoms is not None else (
        " Without bulk_atoms= labels count the atoms per cell of this slab, so an "
        "in-plane supercell of a genslab slab has e.g. 'O16' where genslab reports "
        "'O4'; pass bulk_atoms= to label planes by the bulk."
    )

    # ---- Planes allowed to become a surface ----
    # A sub-slab is a contiguous run of planes [bottom, top] of the input
    # slab; it never wraps through the vacuum.
    known_names = set(plane_names) | ({recon_label} if recon_eligible else set())
    if cut_at == "all":
        valid_boundary_names = set(plane_names)
    elif cut_at == "termination":
        # A deformed (primed) surface plane stands for its bulk plane type.
        ends = {plane_names[0].rstrip("'"), plane_names[-1].rstrip("'")}
        valid_boundary_names = {name for name in plane_names if name.rstrip("'") in ends}
    elif isinstance(cut_at, str):
        matched = {name for name in known_names if plane_name_matches(cut_at, name)}
        if not matched:
            raise ValueError(
                f"Plane name {cut_at!r} not found. "
                f"Available: {sorted(set(plane_names))}.{label_hint}"
            )
        valid_boundary_names = matched
    elif isinstance(cut_at, list):
        valid_boundary_names = set()
        unknown = []
        for query in cut_at:
            matched = {name for name in known_names if plane_name_matches(query, name)}
            if not matched:
                unknown.append(query)
            else:
                valid_boundary_names |= matched
        if unknown:
            raise ValueError(
                f"Unknown plane names: {unknown}. "
                f"Available: {sorted(set(plane_names))}.{label_hint}"
            )
    else:
        raise ValueError(
            f"Invalid cut_at={cut_at!r}. Must be 'all', 'termination', "
            f"a plane name, or list of plane names."
        )
    # Copies of the cut plane become surfaces only if its reconstructed
    # label is among the requested ones.
    exposable = recon_eligible if recon_label in valid_boundary_names else set()
    boundary_indices = sorted(
        {i for i in range(n) if plane_names[i] in valid_boundary_names} | exposable
    )

    # ---- Evaluate every candidate cut on the actual atoms ----
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
            if abs(total_q) > charge_tol * stoich_k:
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

    if (
        reconstruction is None and cut_at == "termination" and n > 2
        and [(c["bottom_plane"], c["top_plane"]) for c in valid_cuts] == [(0, n - 1)]
        and not {plane_names[0].rstrip("'"), plane_names[-1].rstrip("'")}
        & {name.rstrip("'") for name in plane_names[1:-1]}
    ):
        warnings.warn(
            f"cutslab returns only the input slab: its surface planes "
            f"({plane_names[0]}, {plane_names[-1]}) occur nowhere inside it.  For a "
            "reconstructed (Tasker III) slab pass reconstruction=term['reconstruction'] "
            "from generate_slabs_for_miller.",
            stacklevel=2,
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
    widest = int(np.argmax(gaps))
    if widest == len(ordered) - 1 and np.all((frac[:, axis] >= 0.0) & (frac[:, axis] < 1.0)):
        return  # already contiguous inside the cell: keep the coordinates
    start = ordered[(widest + 1) % len(ordered)]
    frac[:, axis] = (f - start) % 1.0
    atoms.set_scaled_positions(frac)


def _merge_surface_planes(planes_sorted, names, recon_counts, atoms_z, charge_tol, max_planes=4):
    """
    Merge the outermost planes at each end of a slab when together they have
    the composition of a reconstructed plane: once atoms are deleted, the
    rest of a rumpled plane may no longer cluster into one plane.  Returns
    the planes and, if given, the matching list of *names*.
    """
    planes = list(planes_sorted)
    names = list(names) if names is not None else None
    for top in (False, True):
        for k in range(2, min(max_planes, len(planes) - 1) + 1):
            group = planes[-k:] if top else planes[:k]
            total = Counter()
            for plane in group:
                total.update(plane["counts"])
            if dict(total) != recon_counts:
                continue
            idx = [i for plane in group for i in plane["indices"]]
            merged = _make_plane(idx, atoms_z[idx, 1], atoms_z, charge_tol)
            if top:
                planes[-k:] = [merged]
                if names is not None:
                    names[-k:] = [names[-1]]
            else:
                planes[:k] = [merged]
                if names is not None:
                    names[:k] = [names[0]]
            break
    return planes, names


def _prepare_reconstruction(reconstruction, cell2d):
    """
    Normalise a reconstruction dict from :func:`generate_slabs_for_miller`
    (also after a JSON round trip: string keys, lists) and express it in the
    in-plane cell *cell2d* of the slab, an integer supercell of the cell the
    pattern was made in.
    """
    if not reconstruction.get("cut_plane_frac"):
        raise ValueError(
            "The reconstruction dict has no 'cut_plane_frac' (made by an older "
            "taskerslabgen); regenerate it with generate_slabs_for_miller."
        )

    def plane(atoms):
        return [(int(a[0]), float(a[1]) % 1.0, float(a[2]) % 1.0) for a in atoms]

    def counts(d):
        return {int(Z): int(c) for Z, c in d.items()}

    ref = plane(reconstruction["cut_plane_frac"])
    delete = plane(reconstruction["delete_info"])
    cut_counts = counts(reconstruction["cut_plane_counts"])
    if reconstruction.get("recon_counts") is not None:
        recon_counts = counts(reconstruction["recon_counts"])
    else:
        recon_counts = Counter(cut_counts)
        recon_counts.subtract(Counter(a[0] for a in delete))
        recon_counts = {Z: c for Z, c in recon_counts.items() if c > 0}
    neighbors = reconstruction.get("neighbor_planes")
    below = plane(neighbors["below"]) if neighbors else None
    above = plane(neighbors["above"]) if neighbors else None
    a3 = reconstruction.get("a3_frac")
    a3 = np.array(a3, dtype=float) if a3 is not None else None

    ref_cell = reconstruction.get("cell2d")
    if ref_cell is not None:
        ref_cell = np.array(ref_cell, dtype=float)
        try:
            M, det = _supercell_matrix(ref_cell, cell2d)
        except ValueError:
            raise ValueError(
                "The slab's in-plane cell is not an integer supercell of the cell the "
                "reconstruction was made in; check that the slab and the reconstruction "
                "come from the same bulk and Miller index."
            ) from None
        if not np.array_equal(M, np.eye(2, dtype=int)):
            ref, delete = _tile_plane(ref, ref_cell, cell2d), _tile_plane(delete, ref_cell, cell2d)
            if neighbors:
                below, above = _tile_plane(below, ref_cell, cell2d), _tile_plane(above, ref_cell, cell2d)
            cut_counts = {Z: c * abs(det) for Z, c in cut_counts.items()}
            recon_counts = {Z: c * abs(det) for Z, c in recon_counts.items()}
            if a3 is not None:
                a3 = a3 @ np.linalg.inv(M)

    remaining = list(delete)
    kept = []
    for atom in ref:
        hit = next((d for d in remaining if d[0] == atom[0]
                    and _frac_distance(d[1:], atom[1:], cell2d) < 1e-3), None)
        if hit is None:
            kept.append(atom)
        else:
            remaining.remove(hit)
    return {
        "recon_label": reconstruction.get("recon_label")
        or f"{reconstruction['cut_plane_name']}-recon",
        "cut_plane_frac": ref,
        "delete_info": delete,
        "kept_frac": kept,
        "cut_plane_counts": cut_counts,
        "recon_counts": recon_counts,
        "below": below,
        "above": above,
        "a3_frac": a3,
        "period": reconstruction.get("period"),
    }


def _reconstruction_deletions(atoms, planes_sorted, outer, recon, cell2d, ab_axes, tol=0.5):
    """
    ``{plane index: atom indices to delete}`` for every interior copy of the
    reconstructed bulk plane.

    A copy must match the bulk plane, and its neighbour planes the bulk
    planes below and above it, after one in-plane translation.  When several
    translations do (a plane that maps onto itself under a non-lattice
    shift), the pattern is chosen as :func:`build_tasker3_slabs` places it:
    the copy ``m`` repeat units away from an already reconstructed surface
    plane (or from the lowest copy) carries the pattern shifted by
    ``m * a3``.
    """
    n = len(planes_sorted)
    frac = atoms.get_scaled_positions()

    def plane_atoms(p):
        return [(int(atoms.numbers[j]), frac[j, ab_axes[0]], frac[j, ab_axes[1]])
                for j in planes_sorted[p]["indices"]]

    def shifts(p, reference):
        found = _plane_translations(reference, plane_atoms(p), cell2d, tol)
        if recon["below"] is None:
            return found
        return [
            t for t in found
            if (p == 0 or _shift_matches(recon["below"], plane_atoms(p - 1), t, cell2d, tol))
            and (p == n - 1 or _shift_matches(recon["above"], plane_atoms(p + 1), t, cell2d, tol))
        ]

    copies = {
        p: shifts(p, recon["cut_plane_frac"]) for p in range(1, n - 1)
        if planes_sorted[p]["counts"] == recon["cut_plane_counts"]
    }
    copies = {p: found for p, found in copies.items() if found}
    if not copies:
        return {}

    anchor, anchor_shifts = None, []
    for i in outer:
        anchor_shifts = shifts(i, recon["kept_frac"])
        if anchor_shifts:
            anchor = i
            break
    if anchor is None:
        anchor = min(copies)
        anchor_shifts = copies[anchor]

    z = [p["z_center"] for p in planes_sorted]
    period, a3 = recon["period"], recon["a3_frac"]

    def expected(T, p):
        """Translation of copy p implied by anchor translation T, and whether
        p is a whole number of repeat units from the anchor."""
        if not period or a3 is None:
            return T, True
        m = (z[p] - z[anchor]) / period
        return T + np.round(m) * a3, abs(m - np.round(m)) * period < min(tol, 0.25 * period)

    def closest(T, p):
        target, whole = expected(T, p)
        return min(copies[p], key=lambda t: _frac_distance(t, target, cell2d)), target, whole

    def consistent(T):
        hits = 0
        for p in copies:
            t, target, whole = closest(T, p)
            hits += whole and _frac_distance(t, target, cell2d) <= tol
        return hits

    T = max(anchor_shifts, key=consistent)  # first of the best, deterministic
    deletions = {}
    for p in sorted(copies):
        t, target, whole = closest(T, p)
        if whole and _frac_distance(t, target, cell2d) > tol:
            raise ValueError(
                f"Plane {p} of the slab is a copy of the reconstructed plane, but its "
                "position does not follow from the slab's reconstructed surface by "
                "whole bulk repeat units; the slab may be too distorted to re-apply "
                "the pattern."
            )
        deletions[p] = _pick_deleted_atoms(atoms, planes_sorted[p]["indices"], recon["delete_info"],
                                           t, frac, ab_axes, cell2d, tol)
    return deletions


def _pick_deleted_atoms(atoms, plane_indices, delete_info, shift, frac, ab_axes, cell2d, tol):
    """Atoms of a plane at the pattern positions *delete_info* + *shift*
    (one-to-one per species, each within *tol* angstrom)."""
    chosen = set()
    plane_indices = np.asarray(plane_indices, dtype=int)
    for Z in sorted({d[0] for d in delete_info}):
        targets = np.array([[d[1], d[2]] for d in delete_info if d[0] == Z]) + shift
        cands = plane_indices[atoms.numbers[plane_indices] == Z]
        d = frac[cands][:, ab_axes][None, :, :] - targets[:, None, :]
        d -= np.round(d)
        dist = np.linalg.norm(d @ cell2d, axis=-1)
        rows, cols = linear_sum_assignment(dist)
        if len(rows) < len(targets) or dist[rows, cols].max() > tol:
            raise ValueError("Could not place the reconstruction pattern on a plane of the slab.")
        chosen.update(int(cands[c]) for c in cols)
    return sorted(chosen)
