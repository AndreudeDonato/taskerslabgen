import numpy as np
from ase.build import surface
from ase.data import atomic_numbers
from ase.io import read
from scipy.optimize import linear_sum_assignment


def build_surface(bulk_atoms, miller, layers=1, vacuum=0.0, verbose=None):
    """
    Build an ASE surface slab from a bulk structure.

    Parameters
    ----------
    bulk_atoms : Atoms
        Bulk unit cell.
    miller : tuple of int
        Miller index ``(h, k, l)``.
    layers : int
        Number of bulk repeat units along the surface normal.
    vacuum : float
        Vacuum to add (angstrom).  Set to 0 for bare slab.
    verbose : bool or None
        Print debug information.

    Returns
    -------
    Atoms
        Surface slab with full PBC enabled.
    """
    slab = surface(bulk_atoms, miller, layers=layers, vacuum=vacuum)
    slab.set_pbc((True, True, True))
    if verbose:
        print("BULK")
        print(bulk_atoms, bulk_atoms.positions, "\n")
        print("REORIENTED BULK")
        print(slab, slab.positions, "\n")
    return slab


def compute_projection(bulk, surf_bulk, charges, miller, verbose=None):
    """
    Compute the z-projection matrix ``[Z, z_coord, charge]`` for each atom.

    Parameters
    ----------
    bulk : Atoms
        Original bulk cell (used to compute the lattice-plane spacing *L*).
    surf_bulk : Atoms
        Reoriented 1-layer surface slab.
    charges : dict or list
        Formal charges.  A dict maps element symbols or atomic numbers to
        charge values; a list gives per-atom charges.
    miller : tuple of int
        Miller index ``(h, k, l)``.
    verbose : bool or None
        Print debug information.

    Returns
    -------
    atoms_z_matrix : ndarray, shape (N, 3)
        Each row is ``[atomic_number, z_position, charge]``.
    L : float
        Lattice-plane spacing (angstrom) for this Miller index.
    """
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


def identify_planes(atoms_z, L, plane_tol=0.05, charge_tol=1e-3):
    """
    Cluster atoms into atomic planes along the stacking direction.

    Atoms whose z-coordinates (mod *L*) differ by less than *plane_tol*
    are grouped into the same plane.  Planes that wrap across the
    periodic boundary are merged.

    Parameters
    ----------
    atoms_z : ndarray, shape (N, 3)
        ``[atomic_number, z_position, charge]`` matrix.
    L : float
        Lattice-plane spacing (angstrom).
    plane_tol : float
        Maximum distance (angstrom) for two atoms to belong to the same
        plane.
    charge_tol : float
        Charges with ``abs(q) < charge_tol`` are set to exactly 0.

    Returns
    -------
    list of dict
        Each dict contains ``z_center``, ``q_total``, ``indices``, and
        ``counts`` (element composition ``{Z: count}``).
    """
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
    """
    Compute the reduced (primitive) stoichiometry of the unit cell.

    Parameters
    ----------
    atoms_z : ndarray, shape (N, 3)
        ``[atomic_number, z_position, charge]`` matrix.

    Returns
    -------
    dict
        ``{atomic_number: reduced_count}`` with the GCD factored out.
    """
    types = np.unique(atoms_z[:, 0].astype(int))
    counts = {Z: int(np.sum(atoms_z[:, 0] == Z)) for Z in types}
    gcd = 0
    for c in counts.values():
        gcd = np.gcd(gcd, c)
    gcd = max(int(gcd), 1)
    reduced = {Z: counts[Z] // gcd for Z in types}
    return reduced


def is_stoichiometric_sequence(sequence_counts, reduced_counts):
    """
    Check whether a plane sequence has an integer multiple of the bulk
    stoichiometry.

    Parameters
    ----------
    sequence_counts : dict
        ``{atomic_number: count}`` for the sequence of planes.
    reduced_counts : dict
        Reduced bulk stoichiometry from :func:`compute_reduced_counts`.

    Returns
    -------
    is_stoich : bool
        True if the sequence is a whole-number multiple of the bulk formula.
    k : int or None
        The multiplier, or None if not stoichiometric.
    """
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
    """
    Enumerate all contiguous plane sequences and compute their charge,
    stoichiometry, and dipole moment.

    Parameters
    ----------
    planes : list of dict
        Plane dicts from :func:`identify_planes`.
    L : float
        Lattice-plane spacing (angstrom).
    reduced_counts : dict
        Reduced bulk stoichiometry.
    charge_tol : float
        Tolerance for charge neutrality.

    Returns
    -------
    list of dict
        Each entry describes a cut sequence with keys ``bottom_cut``,
        ``top_cut``, ``plane_indices``, ``total_charge``, ``net_dipole``,
        ``is_neutral``, ``is_stoich``, ``stoich_k``, etc.
    """
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
    """
    Select the best stoichiometric, charge-neutral sequence (lowest dipole).

    Parameters
    ----------
    sequences : list of dict
        Output of :func:`enumerate_cut_pairs`.
    dipole_tol : float
        Threshold below which the dipole is considered zero (Tasker I/II).

    Returns
    -------
    dict or None
        Best sequence dict with an added ``is_tasker_ii`` flag, or None.
    """
    valid = [s for s in sequences if s["is_neutral"] and s["is_stoich"]]
    if not valid:
        return None
    best = valid[-1]
    best["is_tasker_ii"] = abs(best["net_dipole"]) <= dipole_tol
    return best


def compute_cut_positions(planes, L, bottom_cut_index, top_cut_index):
    """
    Compute z-coordinates for the bottom and top cuts (midpoints between
    adjacent planes).

    Parameters
    ----------
    planes : list of dict
        Plane dicts from :func:`identify_planes`.
    L : float
        Lattice-plane spacing (angstrom).
    bottom_cut_index : int
        Index of the plane *below* the bottom cut.
    top_cut_index : int
        Index of the plane *above* the top cut.

    Returns
    -------
    zbot : float
        z-coordinate of the bottom cut.
    ztop : float
        z-coordinate of the top cut.
    """
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


def apply_vacuum_to_slab(atoms, vacuum=15.0, axis=2):
    """
    Add vacuum above and below a slab by shifting atoms and resizing the cell.

    Atoms are shifted so the bottom of the slab sits at 0 along the given axis,
    and the cell is extended by `vacuum` angstrom on each side (top and bottom).
    """
    if vacuum <= 0:
        return
    positions = atoms.get_positions()
    z_positions = positions[:, axis]
    zmin = float(np.min(z_positions))
    zmax = float(np.max(z_positions))
    new_zmin = zmin - vacuum
    new_zmax = zmax + vacuum
    new_height = new_zmax - new_zmin
    positions[:, axis] -= new_zmin
    atoms.set_positions(positions)
    cell = atoms.get_cell().copy()
    vec = np.zeros(3)
    vec[axis] = new_height
    cell[axis] = vec
    atoms.set_cell(cell)
    atoms.set_pbc([True, True, True])


def assign_plane_names(planes_sorted, atoms=None, axis=2, xy_tol=0.1):
    """
    Assign a type name to each plane based on its elemental composition
    and, when *atoms* is provided, its in-plane spatial arrangement.

    The spatial fingerprint uses each atom's displacement from the
    PBC-aware centroid of the reference species (lowest Z).  This is
    translation-invariant so that equivalent planes at different
    absolute positions in a supercell receive the same name, while
    genuinely different stacking arrangements (e.g. ABAB in fluorite
    110) are correctly distinguished.

    Fingerprints are compared within *xy_tol* (fractional-coordinate
    tolerance) to handle small displacements from relaxation.

    Returns ``(names, name_map)`` where ``names[i]`` is the name of
    ``planes_sorted[i]`` and ``name_map`` is ``{name: counts_dict}``.
    """
    frac_all = None
    ab_axes = None
    if atoms is not None:
        frac_all = atoms.get_scaled_positions()
        ab_axes = [i for i in range(3) if i != axis]

    seen_fps = []
    seen_names = []
    counter = 0
    names = []
    name_map = {}

    for plane in planes_sorted:
        fp = _plane_fingerprint(plane, atoms, frac_all, ab_axes)

        matched_name = None
        for i, sfp in enumerate(seen_fps):
            if _fingerprints_match(fp, sfp, xy_tol):
                matched_name = seen_names[i]
                break

        if matched_name is not None:
            names.append(matched_name)
        else:
            name = f"P{counter}"
            seen_fps.append(fp)
            seen_names.append(name)
            name_map[name] = dict(plane["counts"])
            counter += 1
            names.append(name)

    return names, name_map


def _pbc_mean_1d(values):
    """Circular (PBC-aware) mean of values on the [0, 1) domain."""
    import math
    s = sum(math.sin(2.0 * math.pi * v) for v in values)
    c = sum(math.cos(2.0 * math.pi * v) for v in values)
    if abs(s) < 1e-12 and abs(c) < 1e-12:
        return 0.0
    return (math.atan2(s, c) / (2.0 * math.pi)) % 1.0


def _plane_fingerprint(plane, atoms, frac_all, ab_axes):
    """
    Build a fingerprint: ``(composition_key, xy_displacement_key)``.

    Displacements are measured from the PBC-aware centroid of the
    lowest-Z species and mapped to [0, 1), making the fingerprint
    invariant to rigid translation of the whole plane within the cell
    while avoiding sign ambiguity at the ±0.5 boundary.
    """
    comp_key = tuple(sorted(plane["counts"].items()))
    if atoms is None or frac_all is None:
        return (comp_key, None)

    by_species = {}
    for idx in plane["indices"]:
        Z = int(atoms.numbers[idx])
        fx = float(frac_all[idx, ab_axes[0]])
        fy = float(frac_all[idx, ab_axes[1]])
        by_species.setdefault(Z, []).append((fx, fy))

    ref_Z = min(by_species.keys())
    cx = _pbc_mean_1d([p[0] for p in by_species[ref_Z]])
    cy = _pbc_mean_1d([p[1] for p in by_species[ref_Z]])

    xy_parts = []
    for Z in sorted(by_species):
        shifted = []
        for fx, fy in by_species[Z]:
            dx = (fx - cx) % 1.0
            dy = (fy - cy) % 1.0
            if dx > 1.0 - 1e-9:
                dx = 0.0
            if dy > 1.0 - 1e-9:
                dy = 0.0
            shifted.append((dx, dy))
        xy_parts.append((Z, tuple(sorted(shifted))))

    return (comp_key, tuple(xy_parts))


def _fingerprints_match(fp1, fp2, tol):
    """
    Check if two plane fingerprints match within *tol*.

    Composition must match exactly.  Spatial displacements (in [0, 1))
    are compared element-wise after sorting, with PBC-aware distance.
    """
    if fp1[0] != fp2[0]:
        return False
    if fp1[1] is None and fp2[1] is None:
        return True
    if fp1[1] is None or fp2[1] is None:
        return False
    if len(fp1[1]) != len(fp2[1]):
        return False
    for (Z1, pos1), (Z2, pos2) in zip(fp1[1], fp2[1]):
        if Z1 != Z2 or len(pos1) != len(pos2):
            return False
        for (x1, y1), (x2, y2) in zip(pos1, pos2):
            dx = abs(x1 - x2)
            dx = min(dx, 1.0 - dx)
            dy = abs(y1 - y2)
            dy = min(dy, 1.0 - dy)
            if dx > tol or dy > tol:
                return False
    return True


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


def extract_termination(reference, charges, axis=2, plane_tol=0.05, charge_tol=1e-3):
    """
    Extract termination fingerprints from a reference slab.

    Parameters
    ----------
    reference : Atoms, str/Path, or dict (genslab termination entry)
        - Atoms / file: extract bottom/top fingerprints only.
        - Dict with ``"tasker_type"`` key: also extract reconstruction
          metadata so cutslab can reapply the same Tasker III pattern.

    Returns
    -------
    dict with keys: ``"bottom"``, ``"top"``, ``"reconstruction"``,
    ``"plane_names"``, ``"plane_name_map"``.
    """
    reconstruction = None
    plane_names = None
    plane_name_map = None

    if isinstance(reference, dict) and "tasker_type" in reference:
        if "atoms" in reference:
            slab = reference["atoms"][0]
        elif "slab_atoms" in reference:
            slab = reference["slab_atoms"][0]
        else:
            raise ValueError("Termination dict must contain 'atoms' or 'slab_atoms'")

        recon = reference.get("reconstruction")
        if recon is not None:
            reconstruction = recon
            plane_names = recon.get("plane_names")
            plane_name_map = recon.get("plane_name_map")

        if plane_names is None and "plane_classification" in reference:
            plane_names = reference["plane_classification"].get("plane_names")
            plane_name_map = reference["plane_classification"].get("plane_name_map")

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

    Returns (matches, rmsd).
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
