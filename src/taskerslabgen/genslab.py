from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, chemical_symbols
from ase.io import write

from .core import (
    _INDEX_KEY,
    _charges_to_list,
    _finalize_slab,
    _oriented_bulk,
    _tag_atom_indices,
    build_surface,
    compute_projection,
    identify_planes,
    compute_reduced_counts,
    enumerate_cut_pairs,
    select_best_sequence,
    compute_cut_positions,
    assign_plane_names,
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
        missing = [tid for tid in prefer_plane if tid not in terminations]
        if missing:
            raise ValueError(
                f"No termination with ID {missing} (prefer_plane={prefer_plane!r}); "
                f"available IDs: {sorted(terminations)}."
            )
        return {tid: terminations[tid] for tid in prefer_plane}

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
    surface_supercell=None,
    max_masks=200000,
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
    charges : dict, list or None
        Formal charges.  A dict maps element symbols (or atomic numbers)
        to charge values; a list gives per-atom charges.  ``None`` uses the
        charges stored on *bulk_atoms*: calculator results ``"charges"``
        (e.g. read from an extxyz file), else ``bulk_atoms.get_initial_charges()``.
        Computed charges (Hirshfeld, Bader) rarely sum exactly to zero, so
        they may need a larger *charge_tol*.
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
        Largest |net charge| per formula unit (e) treated as neutral
        (default 1e-3).
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
          the cut, then densest surface planes.  Tasker III: fewest bonds
          broken at the surfaces, then distribution score.
        - ``"all"``: return every candidate, generating a plot for each.
    savecandidates : bool
        Save all valid candidates to an extxyz file for visual inspection.
    surface_supercell : tuple of int or None
        ``(n1, n2)``: repeat the surface cell in-plane before cutting, e.g.
        when a Tasker III plane has an odd excess per surface.  The facet
        stays the same (unlike ``bulk_atoms * (2, 2, 1)``, which changes the
        meaning of the Miller index unless the normal is along c).  Plane
        labels then count the atoms of the supercell (``O16`` for four
        ``O4`` cells).
    max_masks : int
        Largest number of Tasker III deletion patterns to enumerate before
        symmetry reduction (default 200000); larger surface cells raise a
        ``ValueError`` instead of running for hours.

    Returns
    -------
    dict
        Nested dict ``{miller: {plane_id: info}}`` where each ``info``
        dict contains:

        - ``"atoms"`` -- list of ``Atoms`` (one per thickness)
        - ``"tasker_type"`` -- ``"I/II"`` or ``"III"``
        - ``"plane_type"`` -- label of the bottom surface plane (e.g.
          ``"O4"``, ``"O4-recon"``)
        - ``"top_plane_type"`` -- label of the top surface plane; it
          differs from ``plane_type`` for asymmetric terminations.  To cut
          at the same terminations, pass
          ``cutslab(cut_at=[plane_type, top_plane_type])`` (or the default
          ``cut_at="termination"``)
        - ``"plane_counts"`` -- element composition of the cut plane
        - ``"reconstruction"`` -- reconstruction metadata (or None)
        - ``"candidate"`` -- raw scoring dict (Tasker I/II: includes
          ``broken_bonds`` per surface cell, ``broken_bonds_by_pair``
          (e.g. ``{"Ce-O": 8, "Ce-Ce": 12}``) and ``surface_density`` in
          atoms/Å²; Tasker III: ``bond_score``, ``distribution_score``,
          ``multiplicity`` (number of symmetry-equivalent deletion patterns
          it stands for), ...).  IDs are in rank order (ID 0 = best) and
          each ID is a distinct termination: symmetry-equivalent Tasker III
          patterns are reported once.

        Every slab is checked to be stoichiometric, neutral and non-polar;
        a :class:`SlabValidationError` is raised otherwise.  A ``ValueError``
        explains when no non-polar reconstruction exists.
    """
    if candidates not in ("best", "all"):
        raise ValueError(f"candidates must be 'best' or 'all', got {candidates!r}")

    if isinstance(millers, tuple) and len(millers) == 3 and all(isinstance(x, (int, float)) for x in millers):
        millers = [millers]

    opts = _Options(
        layers=tuple(layer_thickness_list), bulk_name=bulk_name, plane_tol=plane_tol,
        charge_tol=charge_tol, dipole_tol=dipole_tol, vacuum=vacuum, plot=plot,
        plot_out_dir=plot_out_dir, verbose=verbose, bond_threshold=bond_threshold,
        bond_distances=bond_distances, max_masks=max_masks,
    )
    return {
        tuple(miller): _generate_for_one_miller(
            bulk_atoms, charges, tuple(miller), opts, prefer_plane, candidates,
            savecandidates, surface_supercell,
        )
        for miller in millers
    }


@dataclass(frozen=True)
class _Options:
    """User options shared by every step of slab generation."""

    layers: tuple
    bulk_name: str = "slab"
    plane_tol: object = None
    charge_tol: float = 1e-3
    dipole_tol: float = 0.05
    vacuum: float = 15.0
    plot: bool = False
    plot_out_dir: str = "."
    verbose: object = None
    bond_threshold: tuple = (0.85, 1.15)
    bond_distances: object = None
    max_masks: int = 200000


@dataclass
class _Facet:
    """One orientation of the bulk, analysed: its planes, labels and charges."""

    bulk: Atoms          # index-tagged bulk (oriented when surface_supercell is used)
    miller: tuple        # Miller index of `bulk` used to build slabs
    out_miller: tuple    # Miller index reported to the user
    surf_bulk: Atoms     # one-layer cell from build_surface
    atoms_z: np.ndarray  # [Z, z, q] of surf_bulk
    L: float             # repeat-unit height along the normal
    planes: list         # planes sorted by z (identify_planes)
    names: list          # plane labels, aligned with `planes`
    name_map: dict
    reduced_counts: dict
    charges_list: list   # charge of every atom of the input bulk

    def validation(self, opts):
        """Keyword arguments of :func:`_finalize_slab`."""
        return {
            "charges_list": self.charges_list,
            "reduced_counts": self.reduced_counts,
            "charge_tol": opts.charge_tol,
            "dipole_tol": opts.dipole_tol,
        }

    def finalize(self, slabs, opts):
        """Validate slabs, drop the index tag, add ``bulk_name``/``miller`` info."""
        for slab in slabs:
            _finalize_slab(slab, **self.validation(opts))
            slab.info["bulk_name"] = opts.bulk_name
            slab.info["miller"] = self.out_miller
        return slabs


def _analyse_facet(bulk_atoms, charges, miller, opts, surface_supercell=None):
    """Planes, labels and charges of the *miller* surface of *bulk_atoms*."""
    # Slabs are cut from an index-tagged copy of the bulk so every slab atom
    # can be traced back to its bulk atom (charges, validation).
    charges_list = _charges_to_list(bulk_atoms, charges)
    if len(charges_list) != len(bulk_atoms):
        raise ValueError(
            f"Charges length ({len(charges_list)}) does not match atoms ({len(bulk_atoms)})."
        )
    bulk = _tag_atom_indices(bulk_atoms)
    out_miller = tuple(miller)
    if surface_supercell is not None:
        bulk = _oriented_bulk(bulk, miller, surface_supercell)
        miller = (0, 0, 1)

    surf_bulk = build_surface(bulk, miller, layers=1, verbose=opts.verbose)
    atoms_z, L = compute_projection(
        bulk, surf_bulk, [charges_list[i] for i in surf_bulk.arrays[_INDEX_KEY]], miller,
        verbose=opts.verbose,
    )
    planes = sorted(
        identify_planes(atoms_z, L, plane_tol=opts.plane_tol, charge_tol=opts.charge_tol),
        key=lambda p: p["z_center"] % L,
    )
    names, name_map = assign_plane_names(planes, atoms=surf_bulk)
    return _Facet(
        bulk=bulk, miller=tuple(miller), out_miller=out_miller, surf_bulk=surf_bulk,
        atoms_z=atoms_z, L=L, planes=planes, names=names, name_map=name_map,
        reduced_counts=compute_reduced_counts(atoms_z), charges_list=charges_list,
    )


def _generate_for_one_miller(bulk_atoms, charges, miller, opts, prefer_plane, candidates,
                             savecandidates, surface_supercell=None):
    h, k, l = miller
    if opts.verbose:
        print(f"\nGenerating Tasker slab for {opts.bulk_name} with Miller index ({h}, {k}, {l})\n")

    facet = _analyse_facet(bulk_atoms, charges, miller, opts, surface_supercell)
    sequences = enumerate_cut_pairs(facet.planes, facet.L, facet.reduced_counts,
                                    charge_tol=opts.charge_tol)
    best_seq = select_best_sequence(sequences, dipole_tol=opts.dipole_tol)
    if best_seq is None:
        raise ValueError("No valid stoichiometry sequences found.")

    if opts.verbose:
        n = len(facet.planes)
        valid_sequences = [s for s in sequences if s["is_neutral"] and s["is_stoich"]]
        print("\nValid stoichiometry sequences (charge-neutral, reduced formula):")
        for i, seq in enumerate(valid_sequences):
            tasker_tag = "Tasker II" if seq["dipole_per_fu"] <= opts.dipole_tol else "Tasker III"
            bottom_edge = f"{seq['bottom_cut']}-{(seq['bottom_cut'] + 1) % n}"
            top_edge = f"{seq['top_cut']}-{(seq['top_cut'] + 1) % n}"
            print(
                f"{i:3d}  dir={seq['direction']}  "
                f"bottom_cut={bottom_edge} top_cut={top_edge}  "
                f"planes={seq['plane_indices']}  Q={seq['total_charge']:+.3f}  "
                f"mu={seq['net_dipole']:+.4e}  "
                f"z_center={seq['z_center']:.3f} {tasker_tag}"
            )

    if best_seq["is_tasker_ii"]:
        return _tasker12_path(facet, sequences, opts, prefer_plane, candidates, savecandidates)

    if opts.verbose:
        print(
            f"No Tasker I/II plane found for miller=({h},{k},{l}) "
            f"on {opts.bulk_name}. Reconstructing Tasker III slab."
        )
    return _tasker3_path(facet, opts, prefer_plane, candidates, savecandidates)


def _select(terminations, prefer_plane, candidates_mode):
    """Apply prefer_plane, then keep the best (lowest ID) if asked."""
    filtered = _filter_by_prefer_plane(terminations, prefer_plane)
    if candidates_mode == "best" and filtered:
        best_tid = min(filtered)  # IDs are in rank order
        return {best_tid: filtered[best_tid]}
    return filtered


def _save_candidates(slabs, facet, opts, kind):
    """Write one slab per candidate to an extxyz file in plot_out_dir."""
    from pathlib import Path

    out_dir = Path(opts.plot_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for tid, slab in enumerate(slabs):
        frame = Atoms(numbers=slab.numbers, positions=slab.positions, cell=slab.cell, pbc=slab.pbc)
        frame.info = {"candidate_id": tid}
        frames.append(frame)
    h, k, l = facet.out_miller
    path = out_dir / f"{opts.bulk_name}_hkl_{h}{k}{l}_candidates.xyz"
    write(path.as_posix(), frames, format="extxyz")
    if opts.verbose:
        print(f"Saved {len(frames)} Tasker {kind} candidates to {path}\n")


def _tasker12_path(facet, sequences, opts, prefer_plane, candidates_mode, savecandidates):
    from .plotting import plot_unitcell_atoms
    from .builder import build_cut_slabs
    from .tasker3 import _bond_pairs, _bonds_across_plane

    planes, names, L = facet.planes, facet.names, facet.L
    n_pl = len(planes)
    h, k, l = facet.out_miller

    # One entry per zero-dipole bulk repeat unit (full period); each gives
    # slabs of exactly `lt` repeat units.  Ranked by bulk bonds broken at the
    # cut, then by surface atom density (denser = more compact), then by
    # label, so the order does not depend on the bulk origin.  IDs follow
    # the ranking (ID 0 = best).
    periodic = Atoms(
        numbers=facet.surf_bulk.numbers, positions=facet.surf_bulk.positions,
        cell=surface_bulk_cell(facet.bulk, facet.miller), pbc=True,
    )
    bonds = _bond_pairs(periodic, opts.bond_threshold, opts.bond_distances)
    area = float(np.linalg.norm(np.cross(facet.surf_bulk.cell[0], facet.surf_bulk.cell[1])))
    ranked = []
    for s in sequences:
        if not (s["is_neutral"] and s["is_stoich"] and s["is_full_period"]
                and s["dipole_per_fu"] <= opts.dipole_tol):
            continue
        cut = s["bottom_cut"]
        bot, top = (cut + 1) % n_pl, cut
        z_cut, _ = compute_cut_positions(planes, L, cut, cut)
        seq = dict(s)
        seq["broken_bonds_by_pair"] = _bonds_across_plane(periodic, z_cut, L, bonds=bonds)
        seq["broken_bonds"] = sum(seq["broken_bonds_by_pair"].values())
        seq["surface_density"] = (
            len(planes[bot]["indices"]) + len(planes[top]["indices"])
        ) / (2.0 * area)
        key = (seq["broken_bonds"], -round(seq["surface_density"], 6), names[bot], names[top], cut)
        ranked.append((key, seq, bot))
    ranked.sort(key=lambda r: r[0])

    terminations = {
        tid: {
            "sequence": seq,
            "plane_type": names[bot],
            "plane_counts": dict(planes[bot]["counts"]),
        }
        for tid, (_, seq, bot) in enumerate(ranked)
    }
    selected = _select(terminations, prefer_plane, candidates_mode)

    def build(seq, layers):
        zbot, ztop = compute_cut_positions(planes, L, seq["bottom_cut"], seq["top_cut"])
        return build_cut_slabs(facet.bulk, facet.miller, layers, zbot, ztop, L, opts.vacuum)

    if savecandidates and ranked:
        _save_candidates([build(seq, [opts.layers[0]])[0] for _, seq, _ in ranked], facet, opts, "I/II")

    output = {}
    for tid, term in selected.items():
        seq = term["sequence"]
        slabs = facet.finalize(build(seq, opts.layers), opts)
        if opts.plot:
            zbot, ztop = compute_cut_positions(planes, L, seq["bottom_cut"], seq["top_cut"])
            bp, tp = names[(seq["bottom_cut"] + 1) % n_pl], names[seq["top_cut"]]
            plot_unitcell_atoms(
                facet.atoms_z, L, facet.out_miller,
                out_png=f"{opts.plot_out_dir}/{opts.bulk_name}_hkl_{h}{k}{l}_{bp}_{tp}_{tid}.png",
                plane_tol=opts.plane_tol, planes=planes,
                zbot=zbot, ztop=ztop, dipole=seq["net_dipole"], plane_names=names,
            )
        output[tid] = {
            "atoms": slabs,
            "tasker_type": "I/II",
            "plane_type": term["plane_type"],
            "top_plane_type": names[seq["top_cut"]],
            "plane_counts": term["plane_counts"],
            "reconstruction": None,
            "candidate": seq,
        }
    return output


def _tasker3_candidates(facet, opts, prefer_plane=None):
    """All scored Tasker III candidates (ranked) and the valid ones; raises
    ``ValueError`` if none is valid."""
    from .tasker3 import (
        build_adjacency_matrix,
        find_tasker3_candidates,
        print_adjacency_matrix,
        _select_tasker3_candidates,
    )

    if opts.verbose:
        adj = build_adjacency_matrix(
            facet.surf_bulk, bond_threshold=opts.bond_threshold,
            bond_distances=opts.bond_distances, bulk_atoms=facet.bulk, miller=facet.miller,
        )
        print(f"Adjacency: {int(np.sum(adj)) // 2} bonds (threshold {opts.bond_threshold})")
        print_adjacency_matrix(adj, facet.surf_bulk)

    candidates = find_tasker3_candidates(
        facet.planes, facet.atoms_z, facet.reduced_counts, None, facet.L,
        surf_bulk=facet.surf_bulk, bond_distances=opts.bond_distances,
        charge_tol=opts.charge_tol, verbose=opts.verbose, prefer_plane=prefer_plane,
        plane_names=facet.names, dipole_tol=opts.dipole_tol,
        bulk_atoms=facet.bulk, miller=facet.miller, bond_threshold=opts.bond_threshold,
        min_layers=min(opts.layers), max_masks=opts.max_masks,
    )
    valid = _select_tasker3_candidates(candidates, facet.out_miller, opts.dipole_tol, opts.charge_tol)
    return candidates, valid


def _tasker3_slabs(facet, opts, cand, layers):
    from .tasker3 import build_tasker3_slabs

    return build_tasker3_slabs(
        facet.bulk, facet.miller, layers,
        cut_plane_idx=cand["cut_plane_idx"], deletion_mask=cand["deletion_mask"],
        planes_sorted=facet.planes, atoms_z_matrix=facet.atoms_z, L=facet.L,
        vacuum=opts.vacuum,
    )


def _tasker3_termination(facet, opts, cand, plot_path=None):
    """Validated slabs, reconstruction metadata and plot of one candidate."""
    from .plotting import plot_unitcell_atoms
    from .tasker3 import _reconstruction_metadata

    i = cand["cut_plane_idx"]
    reconstruction = _reconstruction_metadata(
        cand, facet.planes, facet.names, facet.name_map,
        facet.atoms_z, facet.surf_bulk, facet.bulk, facet.miller, facet.L,
    )
    slabs = facet.finalize(_tasker3_slabs(facet, opts, cand, opts.layers), opts)
    if opts.verbose:
        def composition(counts):
            return "+".join(
                f"{v}{chemical_symbols[Z]}" if v > 1 else chemical_symbols[Z]
                for Z, v in sorted(counts.items())
            )

        print(
            f"  {facet.names[i]}={composition(facet.planes[i]['counts'])}  "
            f"-> {cand['recon_label']}={composition(reconstruction['recon_counts'])}  "
            f"mu={cand['net_dipole']:+.4e}  bonds_broken={cand['bond_score']}"
        )
    if opts.plot and plot_path is not None:
        zbot, ztop = compute_cut_positions(facet.planes, facet.L, (i - 1) % len(facet.planes), i)
        recon_names = list(facet.names)
        recon_names[i] = cand["recon_label"]
        plot_unitcell_atoms(
            facet.atoms_z, facet.L, facet.out_miller,
            out_png=plot_path, plane_tol=opts.plane_tol, planes=facet.planes,
            zbot=zbot, ztop=ztop, dipole=cand["net_dipole"], plane_names=recon_names,
        )
    return slabs, reconstruction


def _tasker3_path(facet, opts, prefer_plane, candidates_mode, savecandidates):
    # Ranked independently of prefer_plane, so IDs do not depend on it;
    # prefer_plane only filters.
    _, valid = _tasker3_candidates(facet, opts)
    terminations = {
        tid: {
            "candidate": cand,
            "plane_type": cand["recon_label"],
            "plane_counts": dict(cand["plane_counts"]),
        }
        for tid, cand in enumerate(valid)
    }
    selected = _select(terminations, prefer_plane, candidates_mode)

    if savecandidates:
        _save_candidates([_tasker3_slabs(facet, opts, cand, [opts.layers[0]])[0] for cand in valid],
                         facet, opts, "III")

    h, k, l = facet.out_miller
    output = {}
    for tid, term in selected.items():
        cand = term["candidate"]
        label = cand["recon_label"]
        if opts.verbose:
            print(f"  Termination {tid}:", end="")
        slabs, reconstruction = _tasker3_termination(
            facet, opts, cand,
            plot_path=f"{opts.plot_out_dir}/{opts.bulk_name}_hkl_{h}{k}{l}_{label}_{label}_{tid}.png",
        )
        output[tid] = {
            "atoms": slabs,
            "tasker_type": "III",
            "plane_type": label,
            "top_plane_type": label,
            "plane_counts": dict(cand["plane_counts"]),
            "reconstruction": reconstruction,
            "candidate": cand,
        }
    return output
