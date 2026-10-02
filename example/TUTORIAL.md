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

Planes are labelled by composition per surface cell and by phase: here `O4`
is a plane of four O atoms, and `O4-recon` the same plane after the Tasker
III reconstruction removed half of them.  The same arrangement shifted or
rotated (another stacking) gets primes (`O4'`, `O4''`); a different
arrangement of the same composition gets a letter (`IrO2-a`, `IrO2-b`).
Section 7 explains phases.  `prefer_plane="O"` keeps only terminations whose
surface plane is pure oxygen.

With `candidates="all"` every termination is returned, ranked: ID 0 breaks the
fewest bonds (`term["candidate"]`), which is what `candidates="best"` keeps.

## 3. Cut a thickness series

```python
sub_slabs = cutslab(
    input_structure=thick,
    charges=charges,
    cut_at="termination",
    cuts="top",
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

`cuts="top"` keeps the bottom termination and cuts from the top; every
sub-slab keeps the input's bottom and top planes (section 7).
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
    dipole_tol=0.05,        # a cut over a relaxed surface carries a small dipole
)
```

Every atom is assigned to the nearest bulk plane, so rumpled surface planes
stay whole, and a plane deformed beyond `deform_tol` gets `~` (`O4~`).
`charges=None` reads charges stored on the `Atoms` object instead.

## 6. Plane tolerance

Atoms closer than `plane_tol` (default 0.1 Å) along the normal form one plane.
Use the same value in `generate_slabs_for_miller` and `cutslab`. Lower it only
when two genuine planes sit closer than 0.1 Å (e.g. the buckled O planes of
marcasite (001), 0.07 Å apart).

## 7. Plane phases: shape, relative and absolute

A plane label has two parts: the **arrangement** (which atoms, and how they
sit) and the **phase** (how that arrangement is shifted and rotated in the
crystal).  Think of each plane as a wave: a smooth periodic density along the
surface.  The wave's shape is the arrangement, and where its peaks fall is
the phase.

```
            |--- cell ---|--- cell ---|
plane O      ▁▃█▃▁▁▁▁▁▁▁▁▁▃█▃▁▁▁▁▁▁▁▁    phase 0
plane O'     ▁▁▁▁▁▁▁▃█▃▁▁▁▁▁▁▁▁▁▃█▃▁▁    same shape, shifted: another stacking
plane O2-b   ▁▃█▃▁▃▆▃▁▁▁▁▁▃█▃▁▃▆▃▁▁▁▁    another shape: another arrangement
```

Going up one repeat unit, the crystal adds its own phase step, because the
repeat vector is tilted.  On rutile (110) that step is half a cell:

```
repeat unit 3   ▁▃█▃▁▁▁▁▁▁▁▁    absolute phase 0
repeat unit 2   ▁▁▁▁▁▁▁▃█▃▁▁    absolute phase ½
repeat unit 1   ▁▃█▃▁▁▁▁▁▁▁▁    absolute phase 0
```

All three are the same crystal plane: their **relative** phase (the phase in
the crystal, the crystal's own step removed) is the same, so they share a
label.  Their **absolute** phase (as seen in the slab) alternates.

`selection=` decides how a label picks planes, in `prefer_plane` and `cut_at`:

| `selection` | `"O'"` selects |
|---|---|
| `"relative"` (default) | only `O'`, in any repeat unit: the same crystal plane |
| `"absolute"` (cutslab) | also exactly over the input slab's own surface plane |
| `"shape"` | `O`, `O'`, `O''`, ...: the arrangement in any phase |

`example/plane_phases.py` shows the three cases.  It writes the series of
both selections of every case to one trajectory,
`output_phases/plane_phases.traj` (frame table printed;
open with `--view`, `ase gui <file>` for the top view along the normal or
`ase gui -R -90x <file>` for a side view), and a plot of every cut in
`output_phases/<case>/<selection>/`: the planes the selection allowed as
bottom surface are red, as top surface blue.

1. **Same arrangement, different relative phase (a vs a').**  Anatase
   (101) has four O₂ planes per repeat unit.  Over the correct termination
   the top O sits 0.73 Å above the Ti; over another phase of the same O₂
   plane only 0.15 Å.  `selection="relative"` keeps the termination
   through the whole thickness series; `"shape"` mixes both.  The mixed
   slabs, from `O2 Ti2 O2'` (6 atoms) up, have a polarity of 0.022 /Å at
   every thickness, which passes the `dipole_tol=0.05` that relaxed slabs
   need: the dipole check cannot keep the termination, the phase does.
2. **Same relative phase, different absolute phase (a---a vs a---a').**
   Rutile IrO₂ (110): the top bridging-O row lies exactly over the bottom one
   for 1, 3, 5 layers (`cut_phase_overlap` = 1) and half a cell off for 2,
   4, 6 (`cut_phase_overlap` = 0).  `"relative"` keeps all thicknesses;
   `"absolute"` keeps the ones in phase with the input slab.
3. **A rotation is a phase too.**  The two IrO₂ planes of rutile (001) are
   one arrangement rotated by 90°: `IrO2` and `IrO2'`.  `"relative"` keeps
   slabs ending on the same plane as the input (an even number of planes);
   `"shape"` adds the odd ones, whose top lies exactly over the bottom.

genslab terminations report both surfaces (`plane_type`, `top_plane_type`);
pass both to `cut_at`.  `plane_name_for_filename("O4'")` gives `O4p` for file
names.

## 8. Runnable scripts

| Script | What it shows |
|--------|----------------|
| `example/CeO2_fluorite.py` | Tasker III candidates (headless; `--view` / `--plot` optional) |
| `example/IrO2_rutile.py` | Tasker I/II across several Miller indices |
| `example/NaAlSi3O8_albite.py` | Tasker I/II for albite (NaAlSi₃O₈) over common Miller indices |
| `example/x2supercell_CeO2_fluorite.py` | Full genslab → cutslab tandem on a supercell |
| `example/relaxed_cutslab.py` | `cutslab(bulk_atoms=...)` on a relaxed slab (synthetic, or yours via `--slab/--bulk`) |
| `example/plane_phases.py` | Plane phases: shape, relative and absolute selection, with ASE GUI views |
| `example/batch_unitcell_slabs.py --quick` | Batch path using shipped `bulk_files/` when workbulks are absent |
