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
- **Errors (C9):** ``cutslab`` validates ``cuts`` up front; an empty result
  no longer reports "Unknown cuts mode".
- Single-plane cells: cut midpoints no longer coincide with the plane.
- Miller indices are validated (integers, not (0, 0, 0), reduced).

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
