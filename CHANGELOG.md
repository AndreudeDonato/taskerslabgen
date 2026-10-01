# Changelog

## Unreleased

Bug-fix pass from the 0.3.1 review (see `REVIEW_NOTES.md`).  Several
defaults produced wrong or invalid slabs without an error; results for the
same input can change, and some calls that used to return invalid slabs now
raise.

### Correctness
- **Plane clustering (C1/C2):** ``identify_planes`` uses single-linkage
  clustering of z with a fixed tolerance (``plane_tol=None`` → 0.1 Å, as
  pymatgen), replacing the adaptive tolerance.  The adaptive version merged
  real planes (ZnO/GaN (0001) came out Tasker I/II, rutile (100) Tasker III),
  depended on the bulk origin, and collapsed slabs with vacuum into a single
  plane, so ``cutslab`` returned no thickness series by default.
- **Tasker type from full periods (M2/M4):** the Tasker type and the Tasker
  I/II terminations use only sequences spanning one whole bulk repeat unit.
  ``layer_thickness=n`` now always gives exactly *n* repeat units, and the
  "best" termination is chosen deterministically (lowest plane index) instead
  of by floating-point noise.  ``select_best_sequence`` returns a copy.
- **Validated output (C3):** every slab returned by
  ``generate_slabs_for_miller``, ``cutslab`` and ``reconstruct_tasker_iii``
  is checked (stoichiometric, neutral, non-polar, no internal gaps; new
  ``validate_slab`` / ``SlabValidationError``).
- **Tasker III deletions (C3):** surface-plane atoms are found from the cut
  positions and atom indices instead of a hard-coded 0.05 Å z-window, which
  silently skipped deletions (e.g. IrO₂ (100) gave Ir₄O₁₀ with charge −4).
  ``build_tasker3_slabs`` deletes the copies of the chosen atoms, so both
  surfaces carry the same pattern; ``cutslab`` re-applies it by aligning the
  reference plane (``reconstruction["cut_plane_frac"]``, new) to each exposed
  plane.
- **Polar reconstructions rejected (C3/M1):** Tasker III candidates must be
  neutral and have ``|dipole| <= dipole_tol``; ``find_tasker3_candidates``
  marks ``is_neutral``.  If none qualifies (e.g. wurtzite (0001)) a
  ``ValueError`` explains why; odd excesses suggest an in-plane supercell.
- **Adjacency (C4):** ``build_adjacency_matrix(..., bulk_atoms=, miller=)``
  uses the true bulk lattice in the surface frame (new
  ``surface_bulk_cell``); the old code mixed rotated positions with the
  unrotated bulk cell, so bond counts were wrong for every facet except
  (001).  It now returns bond counts over periodic images (``adj > 0`` for
  the old boolean view).  ``bulk_atoms`` without ``miller`` raises.
- **cutslab ``cut_at="all"`` (C5):** uses the same contiguous search as the
  other modes; it no longer glues top and bottom planes across the vacuum,
  returns duplicates, or anchors ``cuts="right"`` at the wrong plane.
- **cutslab Tasker III fallback removed:** it treated a slab as a periodic
  bulk cell and returned non-stoichiometric slabs.  A slab with no
  zero-dipole cut now raises a ``ValueError`` pointing to
  ``generate_slabs_for_miller`` + ``reconstruction=``.  ``bond_threshold`` /
  ``bond_distances`` of ``cutslab`` are unused.
- **Plane labels (C6, N1) — breaking:** planes are labelled by composition
  (``O4``, ``Ce2O4``, ``Ir2O2``; ASE "metal" order) instead of ``P{n}{letter}``.
  The old numbers followed stacking order, so genslab (bulk cell) and
  cutslab (slab) gave the same plane different labels and genslab's
  ``plane_type`` selected the wrong planes in ``cutslab(cut_at=...)``; labels
  also changed with the bulk origin.  Composition labels depend only on the
  plane.  Geometric variants of one composition get ``-a``/``-b`` (e.g.
  rutile (001) ``IrO2-a``/``IrO2-b``) in a translation-invariant order.
  Matching is translation-invariant; ``xy_tol`` is now in Å (default 0.5).
  With ``reconstruction=``, cutslab labels reconstructed surfaces
  ``<label>-recon`` like genslab.  ``prefer_plane="P0"``-style filters must be
  rewritten (e.g. ``"O4"``, ``"IrO2"``).
- **``dipole_tol`` per formula unit (C8, N2) — breaking:** ``dipole_tol`` is now
  the largest |dipole| per formula unit (e·Å) treated as zero, default 0.05
  (was 1e-6 e·Å per cell).  It is applied the same way to the bulk repeat
  unit (classification), Tasker III reconstructions (thick-slab value),
  every ``cutslab`` cut and the final validation.  The old default rejected
  real inputs: IrO₂ (110) from a CIF with 4-decimal coordinates has a
  6e-4 e·Å noise dipole and was classified Tasker III.  The old absolute
  per-cut check in ``cutslab`` also rejected thick cuts of relaxed slabs
  first.  Polar repeat units are ~1–6 e·Å per formula unit; relaxed
  structures typically need ~0.3.  Sequences carry a new ``dipole_per_fu``
  field, Tasker III candidates too.
- **Best Tasker I/II termination (N3):** terminations are ranked by bulk
  bonds broken at the cut (true lattice, same bond rules as Tasker III),
  then by surface atom density, then by label; ``candidates="all"`` IDs
  follow the ranking.  The old "first in stacking order" changed with the
  bulk origin (e.g. albite (001) picked the Si cut for one origin and the
  O cut for others; the Si cut breaks 4 bonds, the O cut 9).  Candidates
  report ``broken_bonds`` and ``surface_density``.
- **cutslab surface descriptor matched to the bulk (new ``bulk_atoms=``,
  ``miller=``, ``deform_tol=``):** with the bulk given, every atom of the slab
  is assigned to the nearest bulk plane (registry learned from the bulk-like
  interior, species-aware), so relaxed surface planes that rumple or shift
  stay whole.  Planes get their bulk label (``O4``) when they match the bulk
  plane within ``deform_tol`` (RMSD after the best rigid shift, default
  0.3 Å) and a primed label (``O4'``) when they are more deformed or changed
  composition; termination matching ignores the prime.  Integer in-plane
  supercells of the bulk cell are supported.  Without the bulk, a 0.15 Å
  rumpling split the surface plane and cutslab returned only the full slab.
- **cutslab checks every cut on the real atoms:** stoichiometry, charge and
  dipole are computed from the actual (possibly relaxed) atoms of the
  sub-slab after reconstruction deletions, instead of plane centres and
  bookkeeping of deleted charges.
- **Tasker III distribution score (M3):** pairs not listed in
  ``bond_distances`` now contribute their Coulomb energy ``q_i q_j / d_ij``
  (with the given charges) instead of ``|d - d_covalent|``, which favoured
  clustered rows: a half-occupied CeO₂ (001) O plane now gets the
  checkerboard by default, as it already did with ``{"O-O": None}``.
- **Broken bonds by pair:** Tasker I/II candidates report
  ``broken_bonds_by_pair`` (e.g. ``{"Ce-O": 8, "Ce-Ce": 12}``).  The default
  covalent-radius rule is kept because some oxides have genuine metal–metal
  bonds (rutile IrO₂), but it also counts non-bonded cation contacts (Ce–Ce
  in CeO₂); the breakdown shows when to set ``bond_distances``.
- **cutslab on wrapped slabs (M9):** a slab that straddles the cell
  boundary along the normal (e.g. centred at z = 0 and wrapped) is first
  shifted so its vacuum sits at the cell boundary; before, planes were
  ordered from the middle of the slab and the series was incomplete.
- **Errors (C9):** ``cutslab`` validates ``cuts`` up front; an empty result
  no longer reports "Unknown cuts mode".
- Single-plane cells: cut midpoints no longer coincide with the plane.
- Miller indices are validated (integers, not (0, 0, 0), reduced).

### Second review pass
- **No internal-gap check in the generators:** deleting atoms from a rumpled
  Tasker III surface plane legitimately widens a gap, so valid slabs were
  rejected (CeO₂ (001) from a bulk with 1e-5 Å noise failed half the time;
  ideal corundum (111)).  ``validate_slab(max_gap=...)`` keeps the option.
- **``charge_tol`` per formula unit everywhere** (cut sequences, cutslab,
  validation), like ``dipole_tol``; a supercell with 1e-4 e per atom of
  charge noise was rejected while its unit cell was accepted.
- **``charges=None``** reads per-atom charges stored on the ``Atoms``
  (calculator results ``"charges"``, else ``initial_charges``).
- **Cuts in the gap between planes:** cut positions lie in the middle of
  the empty gap between the outermost atoms of neighbouring planes, not
  between plane centres, which could slice a thick plane (albite (-1,1,-1)
  with ``plane_tol=0.2`` gave a slab with a 2L dipole).  Sequence dipoles
  include each plane's internal dipole (planes carry ``z_lo``, ``z_hi``,
  ``dipole``).
- **cutslab on slabs with ``FixAtoms``:** the vacuum shift ignored fixed
  atoms and every cut failed; slabs already inside their cell keep their
  coordinates (``vacuum=0`` returns parent coordinates again).
- **Exact in-plane tiling** for bulk matching of oblique supercells (failed
  with 1e-6 relative cell noise).
- ``prefer_plane=<id>`` with an unknown ID raises instead of returning
  nothing; ``plot=True`` creates ``plot_out_dir``; ``parse_hirshfeld_fhi_aims``
  returns the last Hirshfeld analysis instead of concatenating all of them;
  cutslab's unknown-label error explains supercell labels.
- **Planes no longer straddle the cell boundary (``build_surface``):**
  ``ase.build.surface`` keeps each atom's coordinate along the oblique bulk
  vector in [0, 1), so a plane crossing z = 0 had its two halves taken from
  different layers and a sheared in-plane geometry.  Labels, the bulk
  plane catalog and reconstruction patterns read that geometry, so they
  changed with the bulk origin (albite (0,0,1) swapped ``O3-a``/``O3-b``).
  ``build_surface`` now moves atoms by bulk lattice vectors so the cell
  boundary lies in the widest atom-free gap along the normal; with vacuum
  the slab is centred.  Positions differ from earlier versions by lattice
  translations and one shift along z.
- **Stable variant letters:** ``-a``/``-b`` are ordered by a
  translation-invariant fingerprint (structure-factor magnitudes and
  triplet phases) compared with a tolerance, instead of rounded fractional
  coordinates; 0.005 Å of noise swapped the letters of rutile (001) planes.
- **Exact Tasker III scoring:** the dipole and charge of each deletion
  pattern are computed from the atoms that remain (closed form over all
  thicknesses), not from plane centres, which ignored which atoms were
  deleted: on rumpled planes every pattern scored zero and the built slab
  was polar (corundum (111) raised although half the patterns are valid).
  A pattern is accepted when it stays neutral and non-polar per formula
  unit for every thickness from the thinnest requested one up.  Valid
  patterns are ranked by dangling bonds, i.e. bulk bonds the slab's atoms
  lose at both surfaces, counted on the true bulk lattice (the old score
  read a bond-count matrix as boolean and excluded neighbour planes by
  index, scoring MgO (111) O-terminated as 0 broken bonds), then by the
  distribution score.  Float noise no longer decides the best pattern (it
  did for CeO₂ (001) in 7 of 30 bulk origins).  Candidates report
  ``recon_label``, ``charge_per_fu``, ``is_valid``; ``net_dipole`` is the
  dipole of the thinnest slab.  ``find_tasker3_candidates`` takes
  ``dipole_tol``, ``bulk_atoms``, ``miller``, ``bond_threshold`` and
  ``min_layers``; its ``adj`` argument is unused.
- **``surface_supercell=(n1, n2)``** for ``generate_slabs_for_miller`` and
  ``reconstruct_tasker_iii`` repeats the surface cell in-plane.  The old
  advice for odd excesses, ``bulk_atoms * (2, 2, 1)``, changes the facet
  whenever the normal is not along c (primitive MgO (111) became (110)).
- Tasker III ``plane_type`` (``O4-recon``) works as ``prefer_plane``; IDs do
  not depend on ``prefer_plane``.  Separate errors for charge and dipole
  failures.
- **cutslab re-applies Tasker III patterns as genslab builds them:** a
  copy of the reconstructed plane is aligned on the plane and on its
  neighbour planes, and among equivalent alignments the one a whole number
  of repeat units (``a3``) from the slab's reconstructed surface is used.
  Before, the first translation found was used, so planes that map onto
  themselves under a non-lattice shift (CeO₂ (001), MgO (111), SrTiO₃
  (110)) got a pattern that depended on atom order and differed from
  genslab's slab of the same thickness; failed alignments silently fell
  back to no shift.  Sub-slabs now equal genslab's slabs, also for shuffled
  atoms.  The metadata gains ``recon_label``, ``recon_counts``,
  ``neighbor_planes``, ``cell2d``, ``a3_frac`` and ``period``; it is
  JSON-serialisable and works for in-plane supercells of the slab (both
  silently returned only the input slab).  A reconstruction that matches
  no plane raises.  Reconstructed surface planes that no longer cluster
  into one plane (corundum (111)) are merged.
- **cutslab ``cut_at`` with ``reconstruction=``:** reconstructed copies are
  cut only if their ``-recon`` label is selected; ``cut_at="Ce2"`` used to
  return ``O4-recon`` slabs.
- **``top_plane_type``** in genslab results: the label of the top surface,
  which differs from ``plane_type`` for asymmetric terminations, so
  ``cut_at=plane_type`` alone failed (albite (0,1,0), IrO₂ (111)).
- cutslab warns when it returns only the input slab because its surface
  planes occur nowhere inside it (a Tasker III slab without
  ``reconstruction=``).
- **cutslab bulk matching on relaxed calculations:** ``bulk_atoms`` may be
  a supercell of the cell the slab was built from (a relaxed 2×2×2 bulk)
  and differ from it by a few per cent of strain; the slab's period along
  the normal is fitted on its interior; when relaxation splits every plane
  (thin rutile (001) slabs), single atoms are registered instead of planes.
  Tested on 48 relaxed FHI-aims slabs of rutile, anatase and fluorite
  oxides, all of which failed or gave partial series before.
- Anatase (101), reported against 0.3.1 (charged Ti₁₄O₂₆ slabs), gives
  stoichiometric, neutral slabs; regression test added
  (``bulk_files/TiO2_anatase.cif``).

### Symmetry-distinct Tasker III terminations (C7, M7)
- Deletion patterns related by a symmetry operation of the crystal that
  keeps the stacking direction (in-plane rotations and mirrors,
  translations including centring, screw axes) give the same slab; only
  one per set is scored, and ``candidate["multiplicity"]`` counts the set.
  Equivalent planes of the cell are merged too.  Tasker III IDs are now
  distinct terminations (CeO₂ (001): 3 instead of 16; a 2×2 cell: 255
  instead of 25 880 patterns scored).  The reduction keeps every distinct
  score (checked against full enumeration).
- ``max_masks`` (default 200 000) on ``generate_slabs_for_miller``,
  ``reconstruct_tasker_iii`` and ``find_tasker3_candidates`` raises before
  enumerating more patterns than that.

### Compatibility
- ``ase.build.surface`` is no longer called with ``vacuum=0`` (deprecated in
  ASE 3.29, slated to raise); ``build_surface`` without vacuum keeps atom
  positions and sets a normal third vector of height ``layers * L``.
- ``ase>=3.22``; pytest turns ``FutureWarning`` into errors.

### Tests
- ``tests/test_regressions.py``: known Tasker types, origin invariance,
  validity of every generated slab, cutslab series (Tasker I/II and III),
  adjacency against a thick-slab reference, plane-label invariance and
  genslab/cutslab label agreement, error messages, no ASE ``FutureWarning``.

## 0.3.1

### Plane naming
- Hierarchical labels ``P{n}{letter}`` (e.g. ``P0a`` / ``P0b``): same
  composition + D4-congruent in-plane geometry share a type; variants
  get letters in stacking order.
- ``prefer_plane="P0"`` / ``cut_at="P0"`` match all variants of that type
  (including ``*-recon``).  Helpers: ``plane_name_base``, ``plane_name_matches``.
- IrO₂ (001) diagonal/anti-diagonal planes are now ``P0a``/``P0b`` instead
  of unrelated ``P0``/``P1`` (downstream ``plane_type`` strings change).

### Adaptive plane clustering
- ``identify_planes(plane_tol=None)`` (new default) infers the merge
  tolerance from the z-gap distribution: coplanar oxide layers still
  merge; staggered silicates (e.g. albite) keep finer atomic cuts.
- Explicit ``plane_tol=<float>`` remains a fixed override.
- Public APIs (``generate_slabs_for_miller``, ``cutslab``,
  ``reconstruct_tasker_iii``) default to adaptive clustering.

### Packaging
- Enrich `pyproject.toml` with license, classifiers, URLs, and `dev` extras.
- Require modern pip/setuptools for PEP 660 editable installs
  (`python3 -m pip install -U pip setuptools wheel`).
- Remove leftover broken `UNKNOWN.egg-info` artifacts from failed installs.
- Bump package version to 0.3.1.

### UX defaults
- Default `plot=False` for `generate_slabs_for_miller`, `cutslab`, and
  `reconstruct_tasker_iii` (quiet library mode; no filesystem plot side effects).
- Gate Tasker III / fallback status prints behind `verbose=True`.

### API / docs
- Document primary vs advanced API in package `__init__` and README.
- Document Tasker III `cut_at="all"` → `"termination"` limitation.
- Add `example/TUTORIAL.md` (genslab → cutslab walkthrough).
- Make README self-contained (no broken out-of-repo links).

### Examples
- Headless examples by default; optional `--view` / `--plot`.
- Batch script falls back to shipped `bulk_files/*.cif` when
  `workbulkfiles/unitcell/*.out` are absent.
- Update `BATCH_SLABS.md` for the new discovery / quick-run path.

### Tests / CI
- Broader smoke coverage (IrO2 Tasker I/II, stoichiometry / dipole checks).
- Add GitHub Actions workflow for pytest on Python 3.10–3.12.
- Tests for hierarchical plane naming and `P0` filter matching.
