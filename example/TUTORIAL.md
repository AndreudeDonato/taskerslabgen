# Tutorial: genslab → cutslab

This walkthrough builds a thick non-polar slab, then peels a thickness series
that keeps the same surface termination chemistry.

## 1. Install

```bash
cd taskerslabgen
python3 -m pip install -U pip setuptools wheel
python3 -m pip install -e ".[dev]"
```

If editable install fails on an old system pip, use:

```bash
python3 -m pip install --no-build-isolation -e ".[dev]"
```

## 2. Generate a thick reference slab

```python
from pathlib import Path
from ase.io import read
from taskerslabgen import generate_slabs_for_miller, cutslab

bulk = read("bulk_files/CeO2_fluorite.cif")
charges = {"Ce": 4.0, "O": -2.0}

result = generate_slabs_for_miller(
    bulk_atoms=bulk,
    charges=charges,
    millers=(0, 0, 1),
    layer_thickness_list=[3],
    bulk_name="CeO2",
    vacuum=15.0,
    bond_distances={"Ce-Ce": None, "O-O": None, "Ce-O": 2.35},
    prefer_plane="O",
    candidates="best",
    # plot=False by default (quiet library mode)
)

term = next(iter(result[(0, 0, 1)].values()))
thick = term["atoms"][0]
print(term["tasker_type"], term["plane_type"], len(thick))
```

`generate_slabs_for_miller` classifies the surface as Tasker I/II (zero dipole)
or Tasker III (needs reconstruction). For Tasker III, keep
`term["reconstruction"]` for the next step.

## 3. Cut a thickness series

```python
sub_slabs = cutslab(
    input_structure=thick,
    charges=charges,
    cut_at="termination",
    cuts="right",
    reconstruction=term.get("reconstruction"),
    vacuum=15.0,
)

for slab in sub_slabs:
    print(
        len(slab),
        slab.info["cut_bottom_plane"],
        slab.info["cut_top_plane"],
        slab.info["cut_n_planes"],
    )
```

`cuts="right"` fixes the bottom termination and peels from the top.
Each returned `Atoms` object carries cut metadata in `.info`.

## 4. Tasker III note

When a reconstruction dict is provided, `cut_at="all"` is forced to
`"termination"` so newly exposed interior planes receive the same deletion
pattern. Prefer `cut_at="termination"` explicitly for reconstructed surfaces.

## 5. Runnable scripts

| Script | What it shows |
|--------|----------------|
| `example/CeO2_fluorite.py` | Tasker III candidates (headless; `--view` / `--plot` optional) |
| `example/IrO2_rutile.py` | Tasker I/II across several Miller indices |
| `example/NaAlSi3O8_albite.py` | Tasker I/II for albite (NaAlSi₃O₈) over common Miller indices |
| `example/x2supercell_CeO2_fluorite.py` | Full genslab → cutslab tandem on a supercell |
| `example/batch_unitcell_slabs.py --quick` | Batch path using shipped `bulk_files/` when workbulks are absent |
