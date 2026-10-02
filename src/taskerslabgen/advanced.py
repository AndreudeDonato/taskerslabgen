"""
Lower-level building blocks of taskerslabgen.

The main workflow needs only :func:`taskerslabgen.generate_slabs_for_miller`
and :func:`taskerslabgen.cutslab`.  The functions here are the steps they
are made of, for inspecting intermediate results or extending the pipeline:
orienting the bulk, clustering planes, enumerating cuts, labelling planes,
bonding, Tasker III candidates and slab builders.
"""

from .builder import build_cut_slabs
from .core import (
    apply_vacuum_to_slab,
    assign_plane_names,
    build_surface,
    compute_cut_positions,
    compute_delete_info,
    compute_projection,
    compute_reduced_counts,
    enumerate_cut_pairs,
    identify_planes,
    is_stoichiometric_sequence,
    select_best_sequence,
    surface_bulk_cell,
)
from .plotting import plot_slab, plot_unitcell_atoms
from .tasker3 import (
    build_adjacency_matrix,
    build_tasker3_slabs,
    find_tasker3_candidates,
    print_adjacency_matrix,
)

__all__ = [
    "apply_vacuum_to_slab",
    "assign_plane_names",
    "build_adjacency_matrix",
    "build_cut_slabs",
    "build_surface",
    "build_tasker3_slabs",
    "compute_cut_positions",
    "compute_delete_info",
    "compute_projection",
    "compute_reduced_counts",
    "enumerate_cut_pairs",
    "find_tasker3_candidates",
    "identify_planes",
    "is_stoichiometric_sequence",
    "plot_slab",
    "plot_unitcell_atoms",
    "print_adjacency_matrix",
    "select_best_sequence",
    "surface_bulk_cell",
]
