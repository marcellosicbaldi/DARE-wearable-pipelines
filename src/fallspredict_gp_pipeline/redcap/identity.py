"""Compatibility import for the implementation in dare_wearables.redcap.identity."""
import sys
from dare_wearables.redcap import identity as _implementation

# Both historical paths resolve to the same module and function globals.
sys.modules[__name__] = _implementation
