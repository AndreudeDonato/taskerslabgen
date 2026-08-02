# Changelog

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
