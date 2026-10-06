"""Compatibility import for the implementation in dare_wearables.redcap.fratup."""
import sys
from dare_wearables.redcap import fratup as _implementation

# Both historical paths resolve to the same module and function globals.
sys.modules[__name__] = _implementation
