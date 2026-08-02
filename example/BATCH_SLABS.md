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

1. `workbulkfiles/unitcell/*.out` — relaxed FHI-aims bulks (preferred for production batches)
2. `bulk_files/*.cif` — shipped demo structures (used automatically when no `.out` files are present)

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

## Troubleshooting

- **No bulk inputs found** — add `.out` files under `workbulkfiles/unitcell/`, or rely on shipped `bulk_files/*.cif`.
- **Missing charges** — extend the `CHARGES` dict in `batch_unitcell_slabs.py`.
- **Tasker III cut_at="all"** — the library forces `"termination"` when a reconstruction is active; the batch script already uses `"termination"`.
