"""Stop the examples with a clear message when Python imports an older taskerslabgen."""
from pathlib import Path

import taskerslabgen

NEEDED = (0, 5)


def require_taskerslabgen():
    """Exit unless the imported taskerslabgen is this repository's version (0.5+)."""
    version = getattr(taskerslabgen, "__version__", None)  # added in 0.5
    parts = tuple(int(x) for x in version.split(".")[:2]) if version else (0, 4)
    if parts >= NEEDED:
        return
    where = Path(taskerslabgen.__file__).resolve().parent
    repo = Path(__file__).resolve().parent.parent
    raise SystemExit(
        f"These examples need taskerslabgen {NEEDED[0]}.{NEEDED[1]} or newer, but Python "
        f"imports version {version or '0.4 or older'} from\n    {where}\n"
        f"Install this repository instead:\n"
        f"    python3 -m pip install -e {repo}\n"
        f"or run with  PYTHONPATH={repo / 'src'}  in front of the command."
    )
