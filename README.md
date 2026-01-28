# taskerslabgen

Utilities to generate Tasker I/II slab terminations using an ASE `Atoms` object
and a charge list (one charge per atom). The library:

- projects atoms along the surface normal
- clusters atoms into planes
- enumerates Tasker cut pairs with stoichiometry + charge checks
- plots planes and chosen cut lines
- builds cut slabs for a list of thicknesses

## Diagram

See `docs/images/cutdiagram_IrO2rutile100.png` for a schematic overview.

![Tasker slab generation diagram](docs/images/cutdiagram_IrO2rutile100.png)

## Folder layout

- `src/taskerslabgen/core.py`  
  Core logic: projection, plane clustering, cut enumeration, selection.
- `src/taskerslabgen/plotting.py`  
  Unit-cell z plot with plane charges and cut lines.
- `src/taskerslabgen/builder.py`  
  Builds slabs by cutting a surface for each thickness.
- `src/taskerslabgen/chargeparsers.py`  
  Charge parsing helpers (FHI-aims Hirshfeld).
- `example/genslab.py`  
  Example script that runs the full flow for multiple Miller indices.
- `bulk_files/`  
  Example bulk input files (e.g., `IrO2_rutile.in` `IrO2_rutile.out`).
- `example/output/`  
  Output folder created by the example script (plots and slabs).

## Outputs

The example script writes (into `example/output/`):

- `*_atoms.png`  
  Plot of unit-cell atoms along z with plane charges and cut lines.
- `*_layers_{thickness}.{ext}`  
  Slab structures for each thickness.

File names include the bulk file stem and Miller index.

## Install

From the repo root:

```
pip install -e .
```

## Example use

Edit `example/genslab.py`:

- set `bulk_path`
- provide `charges` (list) or a path to a text file of charges
- set `miller` and `layer_thickness_list`

Then run:

```
python example/genslab.py
```
