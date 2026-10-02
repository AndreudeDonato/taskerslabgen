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
:class:`PolarSurfaceError`, :func:`dipole_per_area` (the polarity
``dipole_tol`` is compared with), the label
helpers :func:`plane_name_matches` / :func:`plane_name_base` /
:func:`plane_name_for_filename`, and
:func:`parse_hirshfeld_fhi_aims`.  Lower-level steps (plane clustering, cut
enumeration, bonding, Tasker III candidates, builders) are in
:mod:`taskerslabgen.advanced`.
"""

import warnings

__version__ = "0.5.1"


def _check_ase_numpy():
    """ASE 3.22 uses ``numpy.product``, which NumPy 2 removed: building any
    surface then fails deep inside ASE.  Say so at import instead."""
    import ase
    import numpy

    def major_minor(version):
        return tuple(int(part) for part in version.split(".")[:2] if part.isdigit())

    if major_minor(numpy.__version__) >= (2, 0) and major_minor(ase.__version__) < (3, 23):
        raise ImportError(
            f"taskerslabgen needs ASE >= 3.23 with NumPy {numpy.__version__} (ASE "
            f"{ase.__version__} uses functions NumPy 2 removed).  Upgrade ASE "
            "(pip install -U ase) or install numpy<2."
        )


_check_ase_numpy()

from . import advanced
from .chargeparsers import parse_hirshfeld_fhi_aims
from .core import (
    PolarSurfaceError,
    SlabValidationError,
    dipole_per_area,
    plane_name_base,
    plane_name_for_filename,
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
    "dipole_per_area",
    "plane_name_matches",
    "plane_name_base",
    "plane_name_for_filename",
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
