import numpy as np
from ase.build import surface
from ase.data import atomic_numbers
from ase.io import read, write


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

    bottom_cut_z, top_cut_z = compute_cut_positions(
        planes, L, best_seq["bottom_cut"], best_seq["top_cut"]
    )

    if verbose:
        valid_sequences = [s for s in sequences if s["is_neutral"] and s["is_stoich"]]
        print("\nValid stoichiometry sequences (charge-neutral, reduced formula):")
        for i, seq in enumerate(valid_sequences):
            tasker_tag = "Tasker II" if abs(seq["net_dipole"]) <= dipole_tol else ""
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
    plot_path = None
    if plot:
        plot_path = f"{plot_out_dir}/{bulk_name}_hkl_{h}{k}{l}_atoms.png"
        plot_unitcell_atoms(
            atoms_z_matrix,
            L,
            miller,
            out_png=plot_path,
            plane_tol=plane_tol,
            planes=planes,
            zbot=bottom_cut_z,
            ztop=top_cut_z,
            dipole=best_seq["net_dipole"],
        )

    if output_ext is None:
        ext = None
        filename_template = None
    else:
        ext = output_ext.lstrip(".")
        filename_template = f"{bulk_name}_hkl_{h}{k}{l}_layers_{{layer_thickness}}.{ext}"
    slabs, slab_paths = build_cut_slabs(
        bulk_atoms=bulk_atoms,
        miller=miller,
        layer_thickness_list=layer_thickness_list,
        zbot=bottom_cut_z,
        ztop=top_cut_z,
        L=L,
        repeat=repeat,
        vacuum=vacuum,
        out_dir=out_dir,
        filename_template=filename_template,
        output_ext=ext,
    )

    return {
        "plot": plot_path,
        "slabs": slab_paths,
        "slab_atoms": slabs,
        "best_sequence": best_seq,
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
    sequences = enumerate_cut_pairs(planes, L, reduced_counts, charge_tol=charge_tol)

    vec = [(1, 0, 0),
           (0, 1, 0),
           (0, 0, 1)]

    miller = vec[axis]
    if plot:
        plot_path = f"{plot_out_dir}/{stem}_axis_{axis}_cut.png"
        plot_unitcell_atoms(
            atoms_z_matrix,
            L,
            miller,
            out_png=plot_path,
            plane_tol=plane_tol,
            planes=planes,
            zbot=None,
            ztop=None,
            dipole=0,
            )  
        
    min_cut = max(seq["bottom_cut"] for seq in sequences)
    sequences.sort(key=lambda x: x['top_cut'])
    valid_sequences = [
        s for s in sequences if s["is_stoich"] and abs(s["net_dipole"]) <= dipole_tol and s["bottom_cut"] == min_cut
    ]


    planes_sorted = sorted(planes, key=lambda p: p["z_center"] % L)
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
    }
