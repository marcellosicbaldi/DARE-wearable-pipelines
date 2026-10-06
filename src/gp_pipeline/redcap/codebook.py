"""Compatibility import for the implementation in dare_wearables.redcap.codebook."""
import sys
from dare_wearables.redcap import codebook as _implementation

# Both historical paths resolve to the same module and function globals.
sys.modules[__name__] = _implementation
