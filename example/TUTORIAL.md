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

Planes are labelled by composition per surface cell: here `O4` is a plane of
four O atoms, and `O4-recon` the same plane after the Tasker III
reconstruction removed half of them. Planes with the same composition but a
different arrangement get a letter (`IrO2-a`, `IrO2-b`). `prefer_plane="O"`
keeps only terminations whose surface plane is pure oxygen.

With `candidates="all"` every termination is returned, ranked: ID 0 breaks the
fewest bonds (`term["candidate"]`), which is what `candidates="best"` keeps.

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
Each returned `Atoms` object carries cut metadata in `.info`, and every
sub-slab is checked to be stoichiometric, neutral and non-polar.

## 4. Tasker III note

When a reconstruction dict is provided, `cut_at="all"` is forced to
`"termination"` so newly exposed interior planes receive the same deletion
pattern. Prefer `cut_at="termination"` explicitly for reconstructed surfaces.

## 5. Relaxed slabs

After relaxing the thick slab (e.g. with FHI-aims), cut the relaxed structure
and pass the bulk it was built from:

```python
relaxed = read("slab_relaxed.out")
sub_slabs = cutslab(
    relaxed,
    charges,
    bulk_atoms=bulk,        # unit cell or supercell, may be relaxed too
    miller=(0, 0, 1),
    reconstruction=term.get("reconstruction"),
    dipole_tol=0.3,         # relaxed surfaces carry small dipoles
)
```

Every atom is assigned to the nearest bulk plane, so rumpled surface planes
stay whole, and a plane deformed beyond `deform_tol` gets a primed label
(`O4'`). `charges=None` reads charges stored on the `Atoms` object instead.

## 6. Plane tolerance

Atoms closer than `plane_tol` (default 0.1 Å) along the normal form one plane.
Use the same value in `generate_slabs_for_miller` and `cutslab`. Lower it only
when two genuine planes sit closer than 0.1 Å (e.g. the buckled O planes of
marcasite (001), 0.07 Å apart).

## 7. Runnable scripts

| Script | What it shows |
|--------|----------------|
| `example/CeO2_fluorite.py` | Tasker III candidates (headless; `--view` / `--plot` optional) |
| `example/IrO2_rutile.py` | Tasker I/II across several Miller indices |
| `example/NaAlSi3O8_albite.py` | Tasker I/II for albite (NaAlSi₃O₈) over common Miller indices |
| `example/x2supercell_CeO2_fluorite.py` | Full genslab → cutslab tandem on a supercell |
| `example/relaxed_cutslab.py` | `cutslab(bulk_atoms=...)` on a relaxed slab (synthetic, or yours via `--slab/--bulk`) |
| `example/batch_unitcell_slabs.py --quick` | Batch path using shipped `bulk_files/` when workbulks are absent |
