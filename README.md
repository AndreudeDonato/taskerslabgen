# taskerslabgen

Utilities to generate stoichiometric non-polar (Tasker I/II/III) slab
terminations from first-principles structures using ASE `Atoms` objects and
formal or computed charges.

Relative to pymatgen cleavage / Tasker-2 half-ion moves, Surfaxe zero-dipole
filters, and Hinuma-style nonpolar slab algorithms, this library emphasises an
ASE-native Tasker I/II→III workflow with scored reconstructions and a
`genslab`→`cutslab` thickness series that preserves termination chemistry.

The library:

- projects atoms along the surface normal for any Miller index
- clusters atoms into planes (single-linkage on z, fixed 0.1 Å tolerance, as in pymatgen)
- enumerates Tasker cut pairs with stoichiometry + charge neutrality + dipole checks
- performs Tasker III surface reconstruction (symmetric deletion, bond scoring,
  Coulomb-like distribution scoring that favours checkerboard arrangements)
- plane labels from composition (`O4`, `Ce2O4`, `IrO2-a`) that are identical in the bulk,
  in cut slabs and for any bulk origin
- cuts thick slabs into thinner sub-slabs preserving termination
- per-cut plots with unique Miller-index-aware filenames
- checks every returned slab: stoichiometric, neutral, non-polar, no internal gaps
  (raises `SlabValidationError` otherwise)

## Diagram

See `docs/images/cutdiagram_IrO2rutile100.png` for a schematic overview.

![Tasker slab generation diagram](docs/images/cutdiagram_IrO2rutile100.png)

## Install

From the repo root:

```bash
python3 -m pip install -U pip setuptools wheel
python3 -m pip install -e ".[dev]"
```

Requires Python >= 3.9, ASE, NumPy, Matplotlib, SciPy.

If editable install fails on an older system `pip`, either upgrade pip or run:

```bash
python3 -m pip install --no-build-isolation -e ".[dev]"
```

## Quick start

Examples are headless by default (write structures; no GUI). Add `--view` or
`--plot` when you want interactive viewing or PNG stacking plots.

### Tasker III reconstructions (CeO2 fluorite)

```bash
python example/CeO2_fluorite.py
```

### Tasker I/II multiple Miller indices (IrO2 rutile)

```bash
python example/IrO2_rutile.py
```

### Silicate example (albite NaAlSi₃O₈)

```bash
python example/NaAlSi3O8_albite.py
```

### Tandem genslab + cutslab (supercell)

```bash
python example/x2supercell_CeO2_fluorite.py
```

### Batch slab generation

```bash
python example/batch_unitcell_slabs.py --quick
```

See [`example/TUTORIAL.md`](example/TUTORIAL.md) for a short narrative walkthrough
and [`example/BATCH_SLABS.md`](example/BATCH_SLABS.md) for batch setup.

---

## Primary API

Start with these two functions:

| Function | Role |
|----------|------|
| `generate_slabs_for_miller` | Classify Tasker I/II vs III and build slabs |
| `cutslab` | Peel a thick slab into a thickness series |

Also commonly useful: `build_adjacency_matrix`, `assign_plane_names`,
`reconstruct_tasker_iii`. Lower-level helpers are re-exported for power users
but are not required for the standard workflow.

Library defaults are quiet: `plot=False` and prints only when `verbose=True`.

Every slab returned by `generate_slabs_for_miller`, `cutslab` and
`reconstruct_tasker_iii` is checked to be stoichiometric, charge-neutral,
non-polar and free of internal gaps; a `SlabValidationError` is raised
otherwise.  When no slab can satisfy the conditions (for example a polar
stacking that symmetric deletion cannot fix, or an odd excess that needs an
in-plane supercell) a `ValueError` explains why instead of returning a polar
or non-stoichiometric structure.

---

## Tandem genslab + cutslab workflow

When working with relaxed supercells, the recommended workflow is:

1. **genslab** — call `generate_slabs_for_miller` on the bulk to produce
   a thick non-polar slab.  This determines the optimal Tasker
   termination (including Tasker III reconstruction if needed).
2. **cutslab** — call `cutslab` on the thick slab with
   `cut_at="termination"` and `cuts="right"` (default).  The bottom
   plane is fixed and the code peels from the top, generating every
   valid thickness down to a single plane.  For Tasker III surfaces,
   pass the `reconstruction` dict from the genslab output so that
   newly exposed interior planes receive the same atomic deletion.

Plane names compare composition and in-plane geometry up to an in-plane
translation, so translated copies of a plane share a name, while variants
related only by a rotation or mirror of the in-plane lattice (ABAB stacking)
get different letters of the same type.

### Tasker III limitation

When a reconstruction pattern is provided, requesting `cut_at="all"` is
forced to `"termination"`. Prefer `cut_at="termination"` explicitly for
reconstructed surfaces so newly exposed planes keep the same deletion mask.

---

## API Reference

### `generate_slabs_for_miller`

```python
from taskerslabgen import generate_slabs_for_miller

result = generate_slabs_for_miller(
    bulk_atoms,
    charges,
    millers,
    layer_thickness_list,
    bulk_name="slab",
    plane_tol=None,
    charge_tol=1e-3,
    dipole_tol=0.05,
    vacuum=15.0,
    plot=False,
    plot_out_dir=".",
    verbose=None,
    bond_threshold=(0.85, 1.15),
    bond_distances=None,
    prefer_plane=None,
    candidates="best",
    savecandidates=False,
)
```

Generate non-polar slabs for one or more Miller indices.  Automatically
classifies each surface as Tasker I/II (zero dipole) or Tasker III
(requires reconstruction).

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `bulk_atoms` | `Atoms` | *required* | Bulk unit cell (ASE `Atoms` object). |
| `charges` | `dict` or `list` | *required* | Formal charges.  Dict maps element symbols (e.g. `{"Ce": 4.0, "O": -2.0}`) or atomic numbers to values; list gives per-atom charges. |
| `millers` | `tuple` or `list[tuple]` | *required* | Single Miller index `(h, k, l)` or list of Miller indices. |
| `layer_thickness_list` | `list[int]` | *required* | Slab thicknesses in bulk repeat units (e.g. `[2, 4, 6]`); a Tasker I/II slab of thickness *n* contains exactly *n* bulk repeat units. |
| `bulk_name` | `str` | `"slab"` | Label used in plot and output filenames. |
| `plane_tol` | `float` or `None` | `None` | Largest z-gap (Å) between neighbouring atoms of one plane (single-linkage clustering). `None` = 0.1 Å. |
| `charge_tol` | `float` | `1e-3` | Tolerance for charge neutrality of a cut sequence. |
| `dipole_tol` | `float` | `0.05` | Largest \|dipole\| per formula unit (e·Å) treated as zero: below it the surface is Tasker I/II, and Tasker III reconstructions must also stay below it. Polar repeat units are ~1–6 e·Å per formula unit; relaxed bulks may need ~0.3. |
| `vacuum` | `float` | `15.0` | Vacuum (Å) added to each side of the slab. |
| `plot` | `bool` | `False` | Generate stacking-axis plots showing planes and cuts. |
| `plot_out_dir` | `str` | `"."` | Directory for output plots. |
| `verbose` | `bool` or `None` | `None` | Print detailed information (plane sequences, candidates, etc.). |
| `bond_threshold` | `tuple[float, float]` | `(0.85, 1.15)` | `(lo, hi)` scaling factors applied to the bond reference distance for the adjacency matrix. Only affects Tasker III. |
| `bond_distances` | `dict` or `None` | `None` | Per-pair bond reference distances (see below). |
| `prefer_plane` | see below | `None` | Plane-type filter applied before candidate selection. |
| `candidates` | `str` | `"best"` | `"best"` or `"all"` (see below). |
| `savecandidates` | `bool` | `False` | Save all valid candidates to an extxyz file. |

**`bond_distances` format**

Keys are `"X-Y"` strings (order irrelevant), e.g. `"Ce-O"`.  Values are either:
- `float` — reference distance (Å), scaled by `bond_threshold` to determine bonding
- `None` — forbid that pair entirely (no bond is created between those elements)

Example:
```python
bond_distances={"Ce-Ce": None, "O-O": None, "Ce-O": 2.35}
```

**`prefer_plane` options**

| Value | Behaviour |
|---|---|
| `None` | No filtering — all terminations returned. |
| `int` (e.g. `0`) | Keep only the termination with that numeric ID. |
| `list[int]` (e.g. `[0, 2]`) | Keep terminations with those IDs. |
| `str` element (e.g. `"O"`) | Keep terminations whose cut plane consists **exclusively** of that element. `"O"` matches pure-O planes but NOT mixed CeO planes. |
| `str` plane label (e.g. `"O4"`) | Keep terminations whose plane label matches. `"IrO2"` matches `IrO2-a` / `IrO2-b` / `IrO2-a-recon`; `"IrO2-a"` matches `IrO2-a` and `IrO2-a-recon`. |
| `list[str]` (e.g. `["O", "Ce"]`) | Keep terminations matching **any** entry. Each element match is exclusive — `["O", "Ce"]` keeps pure-O OR pure-Ce planes but not mixed CeO. |

**`candidates` options**

| Value | Behaviour |
|---|---|
| `"best"` | Return only the single best candidate per Miller index. Tasker I/II: the first zero-dipole bulk repeat unit in stacking order (deterministic). Tasker III: lowest `abs_dipole`, then `bond_score`, then `distribution_score`. |
| `"all"` | Return every valid candidate, generating a separate plot for each. |

**Returns**

Nested dict: `{miller_tuple: {plane_id: info_dict}}`.

Each `info_dict` contains:
- `"atoms"` — list of `Atoms` objects (one per thickness)
- `"tasker_type"` — `"I/II"` or `"III"`
- `"plane_type"` — label of the cut plane (e.g. `"O4"`, `"O4-recon"`); `cutslab` uses the same labels, so it can be passed to `cut_at`
- `"plane_counts"` — `{atomic_number: count}` composition of the cut plane
- `"reconstruction"` — reconstruction metadata dict (Tasker III) or `None`
- `"candidate"` — raw scoring dict with dipole, bond score, etc.

Each `Atoms` object carries metadata in `.info`:
- `"bulk_name"` — the `bulk_name` parameter
- `"miller"` — the Miller index tuple

---

### `cutslab`

```python
from taskerslabgen import cutslab

sub_slabs = cutslab(
    input_structure,
    charges,
    axis=2,
    plane_tol=None,
    charge_tol=1e-3,
    dipole_tol=0.05,
    plot_out_dir=".",
    plot=False,
    verbose=None,
    bond_threshold=(0.85, 1.15),
    bond_distances=None,
    reconstruction=None,
    cut_at="termination",
    cuts="right",
    vacuum=15.0,
)
```

Cut an existing thick slab into thinner sub-slabs preserving surface
termination.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_structure` | `Atoms` or path | *required* | Thick slab to cut. |
| `charges` | `dict` or `list` | *required* | Formal charges (same format as `generate_slabs_for_miller`). |
| `axis` | `int` | `2` | Cartesian axis perpendicular to the surface (0=x, 1=y, 2=z). |
| `plane_tol` | `float` or `None` | `None` | Largest z-gap (Å) between neighbouring atoms of one plane (single-linkage clustering). `None` = 0.1 Å. |
| `charge_tol` | `float` | `1e-3` | Tolerance for charge neutrality. |
| `dipole_tol` | `float` | `0.05` | Largest \|dipole\| per formula unit (e·Å) of a sub-slab treated as zero; relaxed slabs usually need ~0.3. |
| `plot_out_dir` | `str` | `"."` | Directory for output plots. |
| `plot` | `bool` | `False` | Generate a stacking-axis plot for each sub-slab. |
| `verbose` | `bool` or `None` | `None` | Print plane stacking and cut details. |
| `bond_threshold` | `tuple[float, float]` | `(0.85, 1.15)` | Unused; kept for backward compatibility. |
| `bond_distances` | `dict` or `None` | `None` | Unused; kept for backward compatibility. |
| `reconstruction` | `dict` or `None` | `None` | Tasker III reconstruction dict from genslab output (`term["reconstruction"]`). When provided, newly exposed interior planes receive the same atomic deletion. Forces `cut_at="termination"` if `cut_at` was `"all"`. |
| `cut_at` | `str` or `list[str]` | `"termination"` | Where to place cuts (see below). |
| `cuts` | `str` | `"right"` | Direction of cuts (see below). |
| `vacuum` | `float` | `15.0` | Vacuum (Å) added to each side of every sub-slab. |

**`cut_at` options**

| Value | Behaviour |
|---|---|
| `"termination"` | Cut only at planes with the labels of the thick slab's top/bottom planes. |
| `"all"` | Cut at any plane that gives a stoichiometric, charge-neutral, zero-dipole sub-slab (contiguous runs of planes only, never across the vacuum). Automatically forced to `"termination"` when `reconstruction` is provided. |
| `str` (e.g. `"O4"`, or genslab's `plane_type`) | Cut at planes whose label matches. `"IrO2"` selects every `IrO2-*` variant; `"IrO2-a"` selects that variant (and `IrO2-a-recon`). |
| `list[str]` (e.g. `["O4", "Ce4"]`) | Cut at planes matching any of the listed labels. |

**`cuts` options**

| Value | Behaviour |
|---|---|
| `"right"` (default) | Fix bottom plane, peel from the top. Produces slabs of decreasing thickness. |
| `"left"` | Fix top plane, peel from the bottom. |
| `"all"` | Keep every valid cut (all combinations of bottom/top boundaries). |

**Returns**

`list[Atoms]` — sub-slabs sorted from smallest to largest by atom count,
each checked to be stoichiometric, neutral, non-polar and free of internal gaps.
If no valid cut exists a `ValueError` says why; for a polar (Tasker III) slab,
build it with `generate_slabs_for_miller` and pass `reconstruction=` instead.

Each `Atoms` object carries metadata in `.info`:
- `"cut_bottom_plane"` — label of the bottom surface plane (`...-recon` if reconstructed)
- `"cut_top_plane"` — label of the top surface plane
- `"cut_bottom_idx"` — integer index of the bottom plane
- `"cut_top_idx"` — integer index of the top plane
- `"cut_n_planes"` — number of atomic planes in the sub-slab

Supports single-plane slabs (1 atomic plane thick).

---

### `build_adjacency_matrix`

```python
from taskerslabgen import build_adjacency_matrix

adj = build_adjacency_matrix(
    atoms,
    bond_threshold=(0.85, 1.15),
    bond_distances=None,
    bulk_atoms=None,
    miller=None,
)
```

Build a bond-count matrix using covalent radii and PBC: `adj[i, j]` is the
number of periodic images of atom *j* bonded to atom *i* (use `adj > 0` for a
boolean view).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `atoms` | `Atoms` | *required* | Structure to compute bonds for. |
| `bond_threshold` | `tuple[float, float]` | `(0.85, 1.15)` | `(lo, hi)` scaling factors on the reference distance. |
| `bond_distances` | `dict` or `None` | `None` | Per-pair reference distances. |
| `bulk_atoms` | `Atoms` or `None` | `None` | Original bulk cell, for `atoms = build_surface(bulk_atoms, miller)`: periodic images then follow the true bulk lattice in the surface frame (`surface_bulk_cell`). |
| `miller` | `tuple` or `None` | `None` | Miller index of `atoms`; required with `bulk_atoms`. |

Returns an `(N, N)` integer `ndarray` (symmetric bond counts).

---

### `assign_plane_names`

```python
from taskerslabgen import assign_plane_names

names, name_map = assign_plane_names(planes_sorted, atoms=None, axis=2, xy_tol=0.5)
```

Label planes by composition (e.g. ``O4``, ``Ce4``, ``Ir2O2``).  A label
depends only on the plane itself, so the same plane gets the same label in the
bulk cell (genslab), in slabs cut from it (cutslab) and for any bulk origin.

- Elements are written metals first, then non-metals, each alphabetically
  (ASE's `"metal"` formula format), e.g. `TiO2`, `SrTiO3`, `Ir2O2`.
- **Variant letter** — when one composition occurs in several geometries not
  related by an in-plane translation, a letter is appended: IrO₂ (001)
  diagonal vs anti-diagonal planes are ``IrO2-a`` / ``IrO2-b``.  Letters
  follow a translation-invariant key of the geometry, not stacking order.
- Reconstructed planes get ``-recon`` (e.g. ``O4-recon``), added by
  genslab/cutslab.

Helpers: `plane_name_base("IrO2-a-recon") == "IrO2"`;
`plane_name_matches("IrO2", "IrO2-a")` is True.  Use these semantics in
`prefer_plane` / `cut_at="IrO2"`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `planes_sorted` | `list[dict]` | *required* | Planes sorted by z-centre. |
| `atoms` | `Atoms` or `None` | `None` | Provide to tell geometric variants apart (otherwise composition only). |
| `axis` | `int` | `2` | Stacking axis. |
| `xy_tol` | `float` | `0.5` | Matching tolerance (Å, in-plane) for corresponding atoms. |

Returns `(names, name_map)` where `names[i]` is the label of
`planes_sorted[i]` and `name_map` is `{label: counts_dict}`.

---

### `reconstruct_tasker_iii`

```python
from taskerslabgen import reconstruct_tasker_iii

result = reconstruct_tasker_iii(
    bulk_atoms, charges, miller, layer_thickness_list, bulk_name,
    plane_tol=None, charge_tol=1e-3, dipole_tol=0.05,
    vacuum=15.0, plot=False, plot_out_dir=".",
    verbose=None, bond_threshold=(0.85, 1.15),
    bond_distances=None, prefer_plane=None,
)
```

Standalone Tasker III reconstruction pipeline.  Can be called directly
when you already know the surface is Tasker III.

Returns a dict with `"slab_atoms"`, `"best_candidate"`,
`"all_candidates"`, `"tasker_type"`, and `"plot"` path.

---

### Advanced helpers

| Function | Description |
|---|---|
| `build_surface(bulk_atoms, miller, layers, vacuum, verbose)` | Build an ASE surface slab from a bulk structure. |
| `compute_projection(bulk, surf_bulk, charges, miller, verbose)` | Compute `[Z, z, q]` matrix and lattice-plane spacing *L*. |
| `identify_planes(atoms_z, L, plane_tol, charge_tol)` | Cluster atoms into atomic planes (single-linkage; `plane_tol=None` = 0.1 Å). |
| `surface_bulk_cell(bulk_atoms, miller)` | True bulk lattice in the frame of `build_surface` (its third vector stacks one layer onto the next). |
| `validate_slab(slab, charges, reduced_counts, ...)` | Check stoichiometry, neutrality, dipole and internal gaps; raises `SlabValidationError`. |
| `compute_reduced_counts(atoms_z)` | Compute reduced (primitive) stoichiometry. |
| `is_stoichiometric_sequence(sequence_counts, reduced_counts)` | Check if a sequence is a whole-number multiple of bulk formula. |
| `enumerate_cut_pairs(planes, L, reduced_counts, charge_tol)` | Enumerate all contiguous plane sequences with charge/dipole info. |
| `select_best_sequence(sequences, dipole_tol)` | Select the best full-period stoichiometric sequence (dipole per formula unit). |
| `compute_cut_positions(planes, L, bottom_cut_index, top_cut_index)` | Compute z-coordinates for bottom and top cuts. |
| `apply_vacuum_to_slab(atoms, vacuum, axis)` | Add vacuum above and below a slab. |
| `compute_delete_info(cut_plane, deletion_mask, atoms_z_matrix, surf_bulk)` | Extract reconstruction deletion pattern as `(Z, fx, fy)` tuples. |
| `extract_termination(reference, charges, axis, plane_tol, charge_tol)` | Extract termination fingerprints from a reference slab. |
| `plane_match_score(plane, ref_fingerprint, atoms, axis)` | Score how well a plane matches a reference fingerprint (Hungarian RMSD). |
| `build_cut_slabs(bulk_atoms, miller, layer_thickness_list, zbot, ztop, L, vacuum)` | Build Tasker I/II slabs at various thicknesses. |
| `plot_unitcell_atoms(atoms_z, L, miller, ...)` | Stacking-axis plot with plane annotations. |
| `parse_hirshfeld_fhi_aims(output_path)` | Parse Hirshfeld charges from an FHI-aims output file. |
| `print_adjacency_matrix(adj, atoms)` | Print adjacency matrix with element labels. |
| `find_tasker3_candidates(planes_sorted, atoms_z_matrix, ...)` | Enumerate and score Tasker III reconstruction candidates. |
| `build_tasker3_slabs(bulk_atoms, miller, ...)` | Build Tasker III slabs with symmetric reconstruction. |

---

## Folder layout

- `src/taskerslabgen/core.py` — shared utilities: projection, plane
  clustering, cut enumeration, composition plane labels, termination
  fingerprinting.
- `src/taskerslabgen/genslab.py` — `generate_slabs_for_miller`.
- `src/taskerslabgen/slabcut.py` — `cutslab`.
- `src/taskerslabgen/tasker3.py` — Tasker III reconstruction: adjacency
  matrix, symmetric deletion, bond/distribution scoring.
- `src/taskerslabgen/plotting.py` — stacking-axis plots.
- `src/taskerslabgen/builder.py` — Tasker I/II slab builder.
- `src/taskerslabgen/chargeparsers.py` — charge parsing (FHI-aims
  Hirshfeld).
- `example/` — runnable example scripts and tutorial.
- `bulk_files/` — example bulk input files.
- `tests/` — smoke tests (`pytest tests/`).

## Outputs

The example scripts write into `example/output*/`:

- `*_hkl_{miller}_cut_{idx}_{bot}_{top}.png` — per-cut plot of atoms
  along z with plane IDs, compositions, charges, and cut boundary lines
  (only when `--plot` is passed).
- Structure files for each slab / sub-slab (CIF by default in demos).

## Running tests

```bash
python3 -m pip install -e ".[dev]"
pytest tests/
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

MIT — see [LICENSE](LICENSE).
