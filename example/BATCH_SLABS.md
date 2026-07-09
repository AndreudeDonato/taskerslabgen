# Batch slab generation (unit-cell bulks)

Generate Tasker I/II/III slabs for many relaxed bulk structures in one run.

**Script:** `example/batch_unitcell_slabs.py`  
**Inputs:** FHI-aims `.out` files in `workbulkfiles/unitcell/`  
**Outputs:** slab `.in` files in `X1output_slabs/` at the repo root

## 1. Clone and install

```bash
git clone https://github.com/AndreudeDonato/taskerslabgen.git
cd taskerslabgen
pip install -e .
```

Requires Python >= 3.9. Dependencies: ASE, NumPy, Matplotlib, SciPy.

## 2. Add bulk structures

Place relaxed unit-cell bulk files here:

```
workbulkfiles/unitcell/
```

Each file must be readable by ASE (FHI-aims `.out` format). The filename stem
is used in output names, e.g. `CeO2_fluorite.out` -> `CeO2_fluorite_hkl_111_cut_32.in`.

Expected stems (16 materials):

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

If you already have these locally from a previous run, copy them in:

```bash
cp /path/to/workbulkfiles/unitcell/*.out workbulkfiles/unitcell/
```

## 3. Run

From the repo root:

```bash
# Full batch (~16 bulks, all Miller indices per crystal type)
python example/batch_unitcell_slabs.py

# Quick smoke test (CeO2 only, one Miller index)
python example/batch_unitcell_slabs.py --quick
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
{bulk_stem}_hkl_{h}{k}{l}_cut_{stoich_k}.in
```

Diagnostic plots are written alongside the slabs in `X1output_slabs/`.

## 5. Customisation

Edit the config block at the top of `example/batch_unitcell_slabs.py`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `OUTPUT_DIR` | `X1output_slabs/` | Where slab files are written |
| `WORKBULKFILES` | `workbulkfiles/unitcell/` | Bulk input directory |
| `THICK_LAYERS` | `6` | Reference slab thickness before cutting |
| `MILLER_BY_CRYSTAL` | per-structure table | Miller indices per polymorph |
| `CHARGES` | metal +4, O -2 | Formal charges for Tasker classification |
| `PREFER_PLANE` | CeO2 (0,0,1) -> O | Force specific plane termination |

## 6. Troubleshooting

| Problem | Fix |
|---------|-----|
| `No .out files found` | Add bulk files to `workbulkfiles/unitcell/` |
| `Charges dict missing entries` | Add the element to `CHARGES` in the script |
| `Unknown crystal type for stem` | Add the stem to `STEM_TO_CRYSTAL` and Miller list to `MILLER_BY_CRYSTAL` |
| `No termination found` | Try relaxing `DIPOLE_TOL` or adjusting `prefer_plane` for that surface |
