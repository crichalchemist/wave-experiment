"""Conftest for tests/scripts/ — ensure project-level scripts/ is importable."""
import sys
from pathlib import Path

# pytest adds tests/ to sys.path which causes tests/scripts/ to shadow
# the project-level scripts/ package.  Fix by ensuring the project root
# is early in sys.path and removing the stale 'scripts' module if it was
# already imported from the wrong location.
_project_root = str(Path(__file__).resolve().parent.parent.parent)

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# If 'scripts' was already imported from tests/scripts, remove it so the
# next import picks up the project-level scripts/ package.
if "scripts" in sys.modules:
    _mod = sys.modules["scripts"]
    if hasattr(_mod, "__file__") and _mod.__file__ and "tests" in _mod.__file__:
        del sys.modules["scripts"]
        # Also remove any sub-modules
        to_remove = [k for k in sys.modules if k.startswith("scripts.")]
        for k in to_remove:
            del sys.modules[k]
