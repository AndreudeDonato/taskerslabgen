"""
taskerslabgen: non-polar oxide slabs, chosen automatically.

Workflow
--------
- :func:`generate_slabs_for_miller` — classify a surface as Tasker I/II or
  III, pick the best termination (or reconstruction) and build slabs
- :func:`cutslab` — cut a thick (relaxed) slab into a thickness series with
  the same terminations
- :func:`reconstruct_tasker_iii` — the Tasker III path on its own

Also here: :func:`validate_slab` / :class:`SlabValidationError`,
:class:`PolarSurfaceError`, the label
helpers :func:`plane_name_matches` / :func:`plane_name_base`, and
:func:`parse_hirshfeld_fhi_aims`.  Lower-level steps (plane clustering, cut
enumeration, bonding, Tasker III candidates, builders) are in
:mod:`taskerslabgen.advanced`.
"""

import warnings

from . import advanced
from .chargeparsers import parse_hirshfeld_fhi_aims
from .core import (
    PolarSurfaceError,
    SlabValidationError,
    plane_name_base,
    plane_name_matches,
    validate_slab,
)
from .genslab import generate_slabs_for_miller
from .slabcut import cutslab
from .tasker3 import reconstruct_tasker_iii

__all__ = [
    "generate_slabs_for_miller",
    "cutslab",
    "reconstruct_tasker_iii",
    "validate_slab",
    "SlabValidationError",
    "PolarSurfaceError",
    "plane_name_matches",
    "plane_name_base",
    "parse_hirshfeld_fhi_aims",
]

_REMOVED = {
    "extract_termination": "use cutslab(cut_at='termination') or the plane labels",
    "plane_match_score": "use plane_name_matches on the plane labels",
}


def __getattr__(name):
    # Names that used to be exported here: still importable, with a warning.
    if name in advanced.__all__:
        warnings.warn(
            f"taskerslabgen.{name} moved to taskerslabgen.advanced.{name}; "
            "the top-level name will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(advanced, name)
    if name in _REMOVED:
        raise AttributeError(f"taskerslabgen.{name} was removed: {_REMOVED[name]}.")
    raise AttributeError(f"module 'taskerslabgen' has no attribute {name!r}")
