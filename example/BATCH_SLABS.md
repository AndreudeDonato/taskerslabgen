# Batch slab generation (unit-cell bulks)

Generate Tasker I/II/III slabs for many bulk structures in one run.

**Script:** `example/batch_unitcell_slabs.py`  
**Outputs:** slab `.in` files in `X1output_slabs/` at the repo root

## 1. Clone and install

```bash
git clone https://github.com/AndreudeDonato/taskerslabgen.git
cd taskerslabgen
python3 -m pip install -U pip setuptools wheel
python3 -m pip install -e ".[dev]"
```

Requires Python >= 3.9. Dependencies: ASE, NumPy, Matplotlib, SciPy.

## 2. Bulk inputs

The script looks for inputs in this order:

1. `--bulk-dir DIR` — every `*.out` / `*.cif` in `DIR`, when given
2. `workbulkfiles/unitcell/*.out` — relaxed FHI-aims bulks (preferred for production batches)
3. `bulk_files/*.cif` — shipped demo structures (used automatically when no `.out` files are present)

Expected stems for a full FHI-aims batch (16 materials):

- `CeO2_fluorite.out`
- `IrO2_rutile.out`
- `OsO2_pyrite.out`
- `OsO2_rutile.out`
- `PbO2_brookite.out`
- `PbO2_rutile.out`
- `PdO2_rutile.out`
- `PtO2_marcasite.out`
- `PtO2_rutile.out`
- `RuO2_rutile.out`
- `SnO2_rutile.out`
- `TiO2_anatase.out`
- `TiO2_rutile.out`
- `VO2_C2m.out`
- `VO2_P2c.out`
- `VO2_rutile.out`

If you already have these locally:

```bash
cp /path/to/workbulkfiles/unitcell/*.out workbulkfiles/unitcell/
```

## 3. Run

From the repo root:

```bash
# Uses shipped CIFs when workbulkfiles are empty
python example/batch_unitcell_slabs.py --quick

# Full batch (all discovered bulks / Miller indices)
python example/batch_unitcell_slabs.py

# Optional stacking plots
python example/batch_unitcell_slabs.py --quick --plot

# Bulks from elsewhere, slabs written elsewhere
python example/batch_unitcell_slabs.py --bulk-dir /path/to/bulks --out-dir /path/to/slabs
```

## 4. What the script does

For each bulk file:

1. **generate_slabs_for_miller** — builds a thick reference slab
   (`THICK_LAYERS = 6` formula units) with the best non-polar termination
   for each Miller index.
2. **cutslab** — cuts the thick slab with `cut_at="termination"` and
   `cuts="right"`, preserving the surface termination (including Tasker III
   reconstruction when needed).

Output files follow:

```
{stem}_hkl_{h}{k}{l}_cut_{stoich_k}.in
```

## 5. Per-material settings

All in the top of `batch_unitcell_slabs.py`:

- `MILLER_BY_CRYSTAL` / `STEM_TO_CRYSTAL` — Miller indices per crystal type.
  Stems not listed get `DEFAULT_MILLERS` (the seven low-index facets).
- `CHARGES` — formal charges by element.
- `BOND_DISTANCES_BY_STEM` — which pairs count as bonds when ranking
  terminations (e.g. CeO2 ignores Ce–Ce contacts).
- `PREFER_PLANE` — force a surface plane for one facet (CeO2 (001): `"O"`).
- `PLANE_TOL` — per-facet plane tolerance, used for both genslab and cutslab.
  The default 0.1 Å absorbs relaxation noise; PtO2 marcasite (001) uses
  0.05 Å because its two O planes 0.07 Å apart allow a better Tasker II cut.
- `DIPOLE_TOL_*` — 0.3 e·Å per formula unit, for relaxed bulks.
- `SELECTION` — `"relative"` (default) keeps the thick slab's surface planes
  exactly (same arrangement and phase); `"shape"` also cuts at shifted or
  rotated copies, adding the slabs that end half a repeat unit off.  See the
  plane phases section of `example/TUTORIAL.md`.
- `DIPOLE_TOL_MAX` — facets with no slab within 0.3 are rebuilt with the
  smallest tolerance that works, up to 1.0, with a warning; cutslab then uses
  the same tolerance.

MoO2 forms Mo–Mo dimers, so a rutile-cell MoO2 has no clean planes: (100),
(010) and (111) need dipole_tol ≈ 0.45 and go through the fallback.

## Troubleshooting

- **No bulk inputs found** — add `.out` files under `workbulkfiles/unitcell/`, or rely on shipped `bulk_files/*.cif`.
- **Missing charges** — extend the `CHARGES` dict in `batch_unitcell_slabs.py`.
- **Short thickness series** — genslab and cutslab must see the same planes;
  give both the same `plane_tol` (the script does, through `PLANE_TOL`).
- **Tasker III cut_at="all"** — the library forces `"termination"` when a reconstruction is active; the batch script already uses `"termination"`.
