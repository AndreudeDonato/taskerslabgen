import math
import warnings
from dataclasses import dataclass, replace

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, chemical_symbols
from ase.io import write

from .core import (
    DEFAULT_DIPOLE_TOL,
    PolarSurfaceError,
    _charge_scale,
    _surface_area,
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
    _repeat_names,
    _surface_a3_xy,
    plane_name_for_filename,
    plane_name_matches,
    surface_bulk_cell,
)


def _filter_by_prefer_plane(terminations, prefer_plane, selection="relative"):
    """
    Filter a {plane_id: info} dict by prefer_plane.

    prefer_plane semantics:
      - ``None``        → no filter (keep everything)
      - ``int``         → keep that single termination ID
      - ``list[int]``   → keep those termination IDs
      - ``str``         → element symbol (e.g. ``"O"``) or plane label,
                           matched by :func:`plane_name_matches` with
                           *selection* (``"O4"`` selects ``O4`` and
                           ``O4-recon``; with ``"shape"`` also ``O4'``).
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
            if any(plane_name_matches(q, pt, selection) for q in plane_type_names):
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
    dipole_tol=DEFAULT_DIPOLE_TOL,
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
    dipole_tol_max=None,
    selection="relative",
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
        Largest |net charge| per formula unit, in units of the mean absolute
        charge per atom, treated as neutral (default 1e-3).
    dipole_tol : float
        Largest polarity still treated as zero (default 1e-3): the |dipole|
        along the normal per surface area, with the charges divided by
        their mean absolute value, in 1/Å (:func:`~taskerslabgen.dipole_per_area`).
        Formal, relative or computed charges then give the same numbers, and
        a slab's polarity does not depend on its thickness.  Every slab
        built (each thickness of *layer_thickness_list*) must stay within
        it; it classifies Tasker I/II and accepts Tasker III
        reconstructions.  Ideal crystals give ~0; a cut surface over a
        relaxed one gives up to ~0.04, a wrong termination of anatase (101)
        0.02, a polar repeat unit ~0.1 per repeat unit.
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
          bottom plane label matches, as *selection* says: ``"O4'"``
          selects ``O4'`` and ``O4'-recon`` (``"relative"``), or every
          phase ``O4``, ``O4'``, ... (``"shape"``).
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
    selection : {"relative", "shape"}
        How a plane label in *prefer_plane* selects planes.  A label is the
        plane's arrangement plus its phase, i.e. how it is shifted and
        rotated in the crystal (``O``, ``O'``, ``O''`` are one arrangement in
        three phases).  ``"relative"`` (default) keeps only that plane;
        ``"shape"`` keeps the arrangement in any phase.  (``"absolute"``
        compares against an existing slab, so it belongs to
        :func:`cutslab`.)
    dipole_tol_max : float or None
        Opt-in fallback for slightly distorted bulks (e.g. relaxed ones that
        lost a symmetry).  When no termination or reconstruction of a facet
        is non-polar within *dipole_tol*, generate the facet again with the
        smallest tolerance that gives a slab (rounded up), if it is at most
        *dipole_tol_max*, and warn.  Facets that succeed with *dipole_tol*
        are not affected.  ``None`` (default) raises
        :class:`~taskerslabgen.PolarSurfaceError` instead.  Same units as
        *dipole_tol*; genuinely polar facets reach ~0.1 per repeat unit, so
        keep it well below that.

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
        - ``"dipole_tol"`` -- the dipole tolerance the slabs were built and
          checked with: *dipole_tol*, or the larger one used by the
          *dipole_tol_max* fallback.  Pass it to :func:`cutslab` to cut
          the same slab
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
        explains when no non-polar reconstruction exists
        (:class:`PolarSurfaceError` when the dipole is what fails).
    """
    if candidates not in ("best", "all"):
        raise ValueError(f"candidates must be 'best' or 'all', got {candidates!r}")
    if selection not in ("relative", "shape"):
        raise ValueError(
            f"selection must be 'relative' or 'shape' here, got {selection!r} "
            "('absolute' compares against an existing slab: use it in cutslab)."
        )
    if dipole_tol_max is not None and dipole_tol_max < dipole_tol:
        raise ValueError(
            f"dipole_tol_max ({dipole_tol_max}) must be at least dipole_tol ({dipole_tol})."
        )

    if isinstance(millers, tuple) and len(millers) == 3 and all(isinstance(x, (int, float)) for x in millers):
        millers = [millers]

    opts = _Options(
        layers=tuple(layer_thickness_list), bulk_name=bulk_name, plane_tol=plane_tol,
        charge_tol=charge_tol, dipole_tol=dipole_tol, vacuum=vacuum, plot=plot,
        plot_out_dir=plot_out_dir, verbose=verbose, bond_threshold=bond_threshold,
        bond_distances=bond_distances, max_masks=max_masks, selection=selection,
    )
    # A plain loop: the fallback's warning points at the caller (stacklevel=3).
    results = {}
    for miller in millers:
        results[tuple(miller)] = _generate_with_dipole_fallback(
            bulk_atoms, charges, tuple(miller), opts, dipole_tol_max, prefer_plane,
            candidates, savecandidates, surface_supercell,
        )
    return results


def _round_up(x, digits=2):
    """*x* rounded up to *digits* significant digits."""
    step = 10.0 ** (math.floor(math.log10(x)) - digits + 1)
    return round(math.ceil(x / step - 1e-9) * step, 12)


def _generate_with_dipole_fallback(bulk_atoms, charges, miller, opts, dipole_tol_max, *args):
    """
    :func:`_generate_for_one_miller`, retried once with the smallest
    dipole tolerance that gives a slab when *dipole_tol_max* allows it.
    Records the tolerance used as ``info["dipole_tol"]``.
    """
    try:
        result = _generate_for_one_miller(bulk_atoms, charges, miller, opts, *args)
    except PolarSurfaceError as exc:
        needed = _round_up(exc.min_dipole * 1.02)
        if dipole_tol_max is None or needed > dipole_tol_max:
            raise
        warnings.warn(
            f"{opts.bulk_name} {miller}: no slab is non-polar within "
            f"dipole_tol={opts.dipole_tol} (the least polar has a polarity of "
            f"{exc.min_dipole:.3g} /A); built with "
            f"dipole_tol={needed}.  Check the bulk's symmetry, and cut these "
            "slabs with cutslab(dipole_tol=info['dipole_tol']).",
            UserWarning,
            stacklevel=3,
        )
        opts = replace(opts, dipole_tol=needed)
        result = _generate_for_one_miller(bulk_atoms, charges, miller, opts, *args)
    for info in result.values():
        info["dipole_tol"] = opts.dipole_tol
    return result


@dataclass(frozen=True)
class _Options:
    """User options shared by every step of slab generation."""

    layers: tuple
    bulk_name: str = "slab"
    plane_tol: object = None
    charge_tol: float = 1e-3
    dipole_tol: float = DEFAULT_DIPOLE_TOL
    vacuum: float = 15.0
    plot: bool = False
    plot_out_dir: str = "."
    verbose: object = None
    bond_threshold: tuple = (0.85, 1.15)
    bond_distances: object = None
    max_masks: int = 200000
    selection: str = "relative"


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
    area: float = 1.0    # surface area of the cell (angstrom^2)
    charge_scale: float = 1.0  # mean absolute charge per atom of the bulk
    repeat: int = 0      # planes in one lattice repeat (fewer than len(planes) for centred cells)

    def validation(self, opts):
        """Keyword arguments of :func:`_finalize_slab`."""
        return {
            "charges_list": self.charges_list,
            "reduced_counts": self.reduced_counts,
            "charge_tol": opts.charge_tol,
            "dipole_tol": opts.dipole_tol,
        }

    def finalize(self, slabs, opts, bottom):
        """Validate slabs, drop the index tag, add ``bulk_name``, ``miller``
        and ``stacking_labels`` (one lattice repeat of plane labels from the
        bottom plane *bottom* up, so cutslab names the planes alike) info."""
        n = len(self.names)
        stacking = [self.names[(bottom + k) % n] for k in range(self.repeat)]
        for slab in slabs:
            _finalize_slab(slab, **self.validation(opts))
            slab.info["bulk_name"] = opts.bulk_name
            slab.info["miller"] = self.out_miller
            slab.info["stacking_labels"] = list(stacking)
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
    a3_xy = _surface_a3_xy(bulk, miller, surf_bulk.cell[:2, :2])
    names, name_map, repeat = _repeat_names(planes, surf_bulk, L, a3_xy)
    return _Facet(
        bulk=bulk, miller=tuple(miller), out_miller=out_miller, surf_bulk=surf_bulk,
        atoms_z=atoms_z, L=L, planes=planes, names=names, name_map=name_map,
        reduced_counts=compute_reduced_counts(atoms_z), charges_list=charges_list,
        repeat=repeat, area=_surface_area(surf_bulk.cell), charge_scale=_charge_scale(atoms_z[:, 2]),
    )


def _generate_for_one_miller(bulk_atoms, charges, miller, opts, prefer_plane, candidates,
                             savecandidates, surface_supercell=None):
    h, k, l = miller
    if opts.verbose:
        print(f"\nGenerating Tasker slab for {opts.bulk_name} with Miller index ({h}, {k}, {l})\n")

    facet = _analyse_facet(bulk_atoms, charges, miller, opts, surface_supercell)
    sequences = enumerate_cut_pairs(facet.planes, facet.L, facet.reduced_counts,
                                    charge_tol=opts.charge_tol, area=facet.area,
                                    charge_scale=facet.charge_scale)
    # A slab of n repeat units has n times the dipole of one: judge by the
    # thickest slab asked for.
    n_units = max(opts.layers)
    best_seq = select_best_sequence(sequences, dipole_tol=opts.dipole_tol, n_units=n_units)
    if best_seq is None:
        raise ValueError("No valid stoichiometry sequences found.")

    if opts.verbose:
        n = len(facet.planes)
        valid_sequences = [s for s in sequences if s["is_neutral"] and s["is_stoich"]]
        print("\nValid stoichiometry sequences (charge-neutral, reduced formula):")
        for i, seq in enumerate(valid_sequences):
            tasker_tag = ("Tasker II" if seq["dipole_per_area"] * n_units <= opts.dipole_tol
                          else "Tasker III")
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
    try:
        return _tasker3_path(facet, opts, prefer_plane, candidates, savecandidates)
    except PolarSurfaceError as exc:
        # The least polar Tasker I/II cut may beat every reconstruction.
        exc.min_dipole = min(exc.min_dipole, best_seq["dipole_per_area"] * n_units)
        raise


def _select(terminations, prefer_plane, candidates_mode, selection="relative"):
    """Apply prefer_plane, then keep the best (lowest ID) if asked."""
    filtered = _filter_by_prefer_plane(terminations, prefer_plane, selection)
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


def _plot_termination(facet, opts, context, slab, bottom, path, title,
                      recon_label=None, removed_tags=()):
    """
    Plot where the bulk was cut (:func:`plot_slab`): *context* is the same
    termination one repeat unit thicker on each side (grey in the plot), so
    the slab sits between the two cuts.  Planes are named like the bulk planes
    they come from, starting at bulk plane *bottom*.  For a reconstruction,
    atoms of the slab's surface planes whose bulk index is in *removed_tags*
    are drawn as removed.  Falls back to plotting *slab* alone if the
    context does not split into whole repeat units.
    """
    from .plotting import plot_slab

    n_pl = len(facet.names)
    n = opts.layers[0]
    atoms_z = np.column_stack([context.numbers, context.positions[:, 2], np.zeros(len(context))])
    planes = sorted(identify_planes(atoms_z, float(context.cell[2, 2]), plane_tol=opts.plane_tol),
                    key=lambda p: p["z_center"])
    extra = 1 if recon_label is not None else 0  # a reconstructed slab ends on the cut plane
    if len(planes) != (n + 2) * n_pl + extra:
        atoms_z = np.column_stack([slab.numbers, slab.positions[:, 2], np.zeros(len(slab))])
        planes = sorted(identify_planes(atoms_z, float(slab.cell[2, 2]), plane_tol=opts.plane_tol),
                        key=lambda p: p["z_center"])
        names = [facet.names[(bottom + k) % n_pl] for k in range(len(planes))]
        if recon_label is not None:
            names[0] = names[-1] = recon_label
        plot_slab(slab, planes, names, path, title=title)
        return
    names = [facet.names[(bottom + k) % n_pl] for k in range(len(planes))]
    lo, hi = n_pl, n_pl + n * n_pl - 1 + extra
    removed = []
    if recon_label is not None:
        for k in (0, lo, hi, len(planes) - 1):
            names[k] = recon_label
        tags = context.arrays[_INDEX_KEY]
        removed = [i for k in (lo, hi) for i in planes[k]["indices"] if tags[i] in set(removed_tags)]
    plot_slab(context, planes, names, path, bottom=lo, top=hi, removed=removed, title=title)


def _tasker12_path(facet, sequences, opts, prefer_plane, candidates_mode, savecandidates):
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
                and s["dipole_per_area"] * max(opts.layers) <= opts.dipole_tol):
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
    selected = _select(terminations, prefer_plane, candidates_mode, opts.selection)

    def build(seq, layers):
        zbot, ztop = compute_cut_positions(planes, L, seq["bottom_cut"], seq["top_cut"])
        return build_cut_slabs(facet.bulk, facet.miller, layers, zbot, ztop, L, opts.vacuum)

    if savecandidates and ranked:
        _save_candidates([build(seq, [opts.layers[0]])[0] for _, seq, _ in ranked], facet, opts, "I/II")

    output = {}
    for tid, term in selected.items():
        seq = term["sequence"]
        slabs = facet.finalize(build(seq, opts.layers), opts, bottom=(seq["bottom_cut"] + 1) % n_pl)
        if opts.plot:
            bottom = (seq["bottom_cut"] + 1) % n_pl
            bp, tp = names[bottom], names[seq["top_cut"]]
            _plot_termination(
                facet, opts, build(seq, [opts.layers[0] + 2])[0], slabs[0], bottom,
                f"{opts.plot_out_dir}/{opts.bulk_name}_hkl_{h}{k}{l}_"
                f"{plane_name_for_filename(bp)}_{plane_name_for_filename(tp)}_{tid}.png",
                f"{opts.bulk_name} ({h}{k}{l}) termination {tid}: {bp} to {tp}",
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
        min_layers=min(opts.layers), max_layers=max(opts.layers), max_masks=opts.max_masks,
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
    from .tasker3 import _reconstruction_metadata

    i = cand["cut_plane_idx"]
    reconstruction = _reconstruction_metadata(
        cand, facet.planes, facet.names, facet.name_map,
        facet.atoms_z, facet.surf_bulk, facet.bulk, facet.miller, facet.L,
    )
    slabs = facet.finalize(_tasker3_slabs(facet, opts, cand, opts.layers), opts, bottom=i)
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
        h, k, l = facet.out_miller
        context = _tasker3_slabs(facet, opts, cand, [opts.layers[0] + 2])[0]
        mask_tags = facet.surf_bulk.arrays[_INDEX_KEY][list(cand["deletion_mask"])]
        _plot_termination(
            facet, opts, context, slabs[0], i, plot_path,
            f"{opts.bulk_name} ({h}{k}{l}) {cand['recon_label']}",
            recon_label=cand["recon_label"], removed_tags={int(t) for t in mask_tags},
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
    selected = _select(terminations, prefer_plane, candidates_mode, opts.selection)

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
            plot_path=(f"{opts.plot_out_dir}/{opts.bulk_name}_hkl_{h}{k}{l}_"
                       f"{plane_name_for_filename(label)}_{plane_name_for_filename(label)}_{tid}.png"),
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
