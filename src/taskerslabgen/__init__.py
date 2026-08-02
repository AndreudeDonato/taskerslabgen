"""
taskerslabgen public API.

Primary entry points (start here)
---------------------------------
- :func:`generate_slabs_for_miller` — classify Tasker I/II vs III and build slabs
- :func:`cutslab` — peel a thick slab into a thickness series

Common helpers
--------------
- :func:`build_adjacency_matrix` — bonding graph for Tasker III scoring
- :func:`assign_plane_names` — stacking-aware labels (``P0a``, ``P0b``, …)
- :func:`reconstruct_tasker_iii` — Tasker III-only path (also used internally)

Advanced / lower-level helpers are also re-exported for power users; prefer the
primary API unless you are extending the pipeline.
"""

from .core import (
    apply_vacuum_to_slab,
    assign_plane_names,
    build_surface,
    compute_cut_positions,
    compute_delete_info,
    compute_projection,
    compute_reduced_counts,
    enumerate_cut_pairs,
    extract_termination,
    identify_planes,
    is_stoichiometric_sequence,
    plane_match_score,
    plane_name_base,
    plane_name_matches,
    select_best_sequence,
)
from .genslab import generate_slabs_for_miller
from .slabcut import cutslab
from .plotting import plot_unitcell_atoms
from .builder import build_cut_slabs
from .chargeparsers import parse_hirshfeld_fhi_aims
from .tasker3 import (
    build_adjacency_matrix,
    build_tasker3_slabs,
    find_tasker3_candidates,
    print_adjacency_matrix,
    reconstruct_tasker_iii,
)

# Primary API (stable workflow surface)
_PRIMARY = (
    "generate_slabs_for_miller",
    "cutslab",
    "build_adjacency_matrix",
    "assign_plane_names",
    "reconstruct_tasker_iii",
)

# Advanced helpers (still public, but lower-level)
_ADVANCED = (
    "build_surface",
    "compute_projection",
    "identify_planes",
    "compute_reduced_counts",
    "enumerate_cut_pairs",
    "select_best_sequence",
    "compute_cut_positions",
    "compute_delete_info",
    "extract_termination",
    "plane_match_score",
    "plane_name_base",
    "plane_name_matches",
    "apply_vacuum_to_slab",
    "is_stoichiometric_sequence",
    "plot_unitcell_atoms",
    "build_cut_slabs",
    "parse_hirshfeld_fhi_aims",
    "print_adjacency_matrix",
    "find_tasker3_candidates",
    "build_tasker3_slabs",
)

__all__ = list(_PRIMARY) + list(_ADVANCED)
