# taskerslabgen

Utilities to generate non-polar (Tasker I/II/III) slab terminations from
first-principles structures using ASE `Atoms` objects and formal or computed
charges. The library:

- projects atoms along the surface normal for any Miller index
- clusters atoms into planes
- enumerates Tasker cut pairs with stoichiometry + charge neutrality + dipole checks
- performs Tasker III surface reconstruction (symmetric deletion, bond scoring,
  Coulomb-like distribution scoring that favours checkerboard arrangements)
- stacking-aware plane naming (composition + fractional-position fingerprints)
- cuts thick slabs into thinner sub-slabs preserving termination
- per-cut plots with unique Miller-index-aware filenames

## Diagram

See `docs/images/cutdiagram_IrO2rutile100.png` for a schematic overview.

![Tasker slab generation diagram](docs/images/cutdiagram_IrO2rutile100.png)

## Install

From the repo root:

```
pip install -e .
```

Requires Python >= 3.9, ASE, NumPy, Matplotlib, SciPy.

## Quick start

### Tasker III reconstructions (CeO2 fluorite)

```
python example/CeO2_fluorite.py
```

### Tasker I/II multiple Miller indices (IrO2 rutile)

```
python example/IrO2_rutile.py
```

### Tandem genslab + cutslab (supercell)

```
python example/x2supercell_CeO2_fluorite.py
```

Example scripts use `Path(__file__)`-relative paths and can be run from any
working directory.

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

Plane identification uses stacking-aware fingerprints (composition +
fractional xy positions relative to the in-plane centroid) so that
ABAB stacking patterns are correctly distinguished.

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
    plane_tol=0.05,
    charge_tol=1e-3,
    dipole_tol=1e-6,
    vacuum=15.0,
    plot=True,
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
| `layer_thickness_list` | `list[int]` | *required* | Slab thicknesses in bulk repeat units (e.g. `[2, 4, 6]`). |
| `bulk_name` | `str` | `"slab"` | Label used in plot and output filenames. |
| `plane_tol` | `float` | `0.05` | Tolerance (Å) for grouping atoms into the same plane. |
| `charge_tol` | `float` | `1e-3` | Tolerance for charge neutrality of a cut sequence. |
| `dipole_tol` | `float` | `1e-6` | Dipole threshold — below this the surface is considered Tasker I/II. |
| `vacuum` | `float` | `15.0` | Vacuum (Å) added to each side of the slab. |
| `plot` | `bool` | `True` | Generate stacking-axis plots showing planes and cuts. |
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
| `str` plane type (e.g. `"P0"`) | Keep terminations whose plane type name matches. `"P0-recon"` also matches `"P0"`. |
| `list[str]` (e.g. `["O", "Ce"]`) | Keep terminations matching **any** entry. Each element match is exclusive — `["O", "Ce"]` keeps pure-O OR pure-Ce planes but not mixed CeO. |

**`candidates` options**

| Value | Behaviour |
|---|---|
| `"best"` | Return only the single best candidate per Miller index (lowest `abs_dipole`, then `bond_score`, then `distribution_score`). |
| `"all"` | Return every valid candidate, generating a separate plot for each. |

**Returns**

Nested dict: `{miller_tuple: {plane_id: info_dict}}`.

Each `info_dict` contains:
- `"atoms"` — list of `Atoms` objects (one per thickness)
- `"tasker_type"` — `"I/II"` or `"III"`
- `"plane_type"` — symbolic plane name (e.g. `"P0"`, `"P0-recon"`)
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
    plane_tol=0.05,
    charge_tol=1e-3,
    dipole_tol=1e-6,
    plot_out_dir=".",
    plot=True,
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
| `plane_tol` | `float` | `0.05` | Tolerance (Å) for grouping atoms into planes. |
| `charge_tol` | `float` | `1e-3` | Tolerance for charge neutrality. |
| `dipole_tol` | `float` | `1e-6` | Dipole threshold for zero-dipole cuts. |
| `plot_out_dir` | `str` | `"."` | Directory for output plots. |
| `plot` | `bool` | `True` | Generate a stacking-axis plot for each sub-slab. |
| `verbose` | `bool` or `None` | `None` | Print plane stacking and cut details. |
| `bond_threshold` | `tuple[float, float]` | `(0.85, 1.15)` | Scaling factors for the adjacency matrix (Tasker III fallback only). |
| `bond_distances` | `dict` or `None` | `None` | Per-pair bond reference distances. |
| `reconstruction` | `dict` or `None` | `None` | Tasker III reconstruction dict from genslab output (`term["reconstruction"]`). When provided, newly exposed interior planes receive the same atomic deletion. Forces `cut_at="termination"` if `cut_at` was `"all"`. |
| `cut_at` | `str` or `list[str]` | `"termination"` | Where to place cuts (see below). |
| `cuts` | `str` | `"right"` | Direction of cuts (see below). |
| `vacuum` | `float` | `15.0` | Vacuum (Å) added to each side of every sub-slab. |

**`cut_at` options**

| Value | Behaviour |
|---|---|
| `"termination"` | Cut only at planes matching the thick slab's top/bottom plane types. |
| `"all"` | Cut at any boundary that gives a stoichiometric, charge-neutral, zero-dipole sub-slab. Automatically forced to `"termination"` when `reconstruction` is provided. |
| `str` (e.g. `"P0"`) | Cut only at boundaries where that plane type is exposed. |
| `list[str]` (e.g. `["P0", "P1"]`) | Cut at boundaries matching any of the listed plane types. |

**`cuts` options**

| Value | Behaviour |
|---|---|
| `"right"` (default) | Fix bottom plane, peel from the top. Produces slabs of decreasing thickness. |
| `"left"` | Fix top plane, peel from the bottom. |
| `"all"` | Keep every valid cut (all combinations of bottom/top boundaries). |

**Returns**

`list[Atoms]` — sub-slabs sorted from smallest to largest by atom count.

Each `Atoms` object carries metadata in `.info`:
- `"cut_bottom_plane"` — plane type name of the bottom surface
- `"cut_top_plane"` — plane type name of the top surface
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
)
```

Build a boolean adjacency matrix using covalent radii and PBC.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `atoms` | `Atoms` | *required* | Structure to compute bonds for. |
| `bond_threshold` | `tuple[float, float]` | `(0.85, 1.15)` | `(lo, hi)` scaling factors on the reference distance. |
| `bond_distances` | `dict` or `None` | `None` | Per-pair reference distances. |
| `bulk_atoms` | `Atoms` or `None` | `None` | Original bulk cell — uses bulk PBC for minimum-image distances. |

Returns an `(N, N)` boolean `ndarray`.

---

### `assign_plane_names`

```python
from taskerslabgen import assign_plane_names

names, name_map = assign_plane_names(planes_sorted, atoms=None, axis=2, xy_tol=0.1)
```

Assign symbolic type names (`P0`, `P1`, ...) to each plane based on
elemental composition and in-plane spatial arrangement.  Stacking-aware:
correctly distinguishes ABAB patterns in e.g. fluorite (110).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `planes_sorted` | `list[dict]` | *required* | Planes sorted by z-centre. |
| `atoms` | `Atoms` or `None` | `None` | Provide to enable spatial fingerprinting. |
| `axis` | `int` | `2` | Stacking axis. |
| `xy_tol` | `float` | `0.1` | Tolerance (fractional coordinates) for fingerprint comparison. |

Returns `(names, name_map)` where `names[i]` is the name of
`planes_sorted[i]` and `name_map` is `{name: counts_dict}`.

---

### `reconstruct_tasker_iii`

```python
from taskerslabgen import reconstruct_tasker_iii

result = reconstruct_tasker_iii(
    bulk_atoms, charges, miller, layer_thickness_list, bulk_name,
    plane_tol=0.05, charge_tol=1e-3, dipole_tol=1e-6,
    vacuum=15.0, plot=True, plot_out_dir=".",
    verbose=None, bond_threshold=(0.85, 1.15),
    bond_distances=None, prefer_plane=None,
)
```

Standalone Tasker III reconstruction pipeline.  Can be called directly
when you already know the surface is Tasker III.

Returns a dict with `"slab_atoms"`, `"best_candidate"`,
`"all_candidates"`, `"tasker_type"`, and `"plot"` path.

---

### Core helpers

| Function | Description |
|---|---|
| `build_surface(bulk_atoms, miller, layers, vacuum, verbose)` | Build an ASE surface slab from a bulk structure. |
| `compute_projection(bulk, surf_bulk, charges, miller, verbose)` | Compute `[Z, z, q]` matrix and lattice-plane spacing *L*. |
| `identify_planes(atoms_z, L, plane_tol, charge_tol)` | Cluster atoms into atomic planes along the stacking direction. |
| `compute_reduced_counts(atoms_z)` | Compute reduced (primitive) stoichiometry. |
| `is_stoichiometric_sequence(sequence_counts, reduced_counts)` | Check if a sequence is a whole-number multiple of bulk formula. |
| `enumerate_cut_pairs(planes, L, reduced_counts, charge_tol)` | Enumerate all contiguous plane sequences with charge/dipole info. |
| `select_best_sequence(sequences, dipole_tol)` | Select the best zero-dipole stoichiometric sequence. |
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
  clustering, cut enumeration, stacking-aware plane naming, termination
  fingerprinting.
- `src/taskerslabgen/genslab.py` — `generate_slabs_for_miller`.
- `src/taskerslabgen/slabcut.py` — `cutslab`.
- `src/taskerslabgen/tasker3.py` — Tasker III reconstruction: adjacency
  matrix, symmetric deletion, bond/distribution scoring.
- `src/taskerslabgen/plotting.py` — stacking-axis plots.
- `src/taskerslabgen/builder.py` — Tasker I/II slab builder.
- `src/taskerslabgen/chargeparsers.py` — charge parsing (FHI-aims
  Hirshfeld).
- `example/` — runnable example scripts.
- `bulk_files/` — example bulk input files.
- `tests/` — smoke tests (`pytest tests/`).

## Outputs

The example scripts write into `example/output*/`:

- `*_hkl_{miller}_cut_{idx}_{bot}_{top}.png` — per-cut plot of atoms
  along z with plane IDs, compositions, charges, and cut boundary lines.
- `*_hkl_{miller}_between_{bot}_{top}_cut_{idx}.{ext}` — slab structure
  files for each sub-slab.

## Running tests

```
pip install pytest
pytest tests/
```

## License

MIT — see [LICENSE](LICENSE).
