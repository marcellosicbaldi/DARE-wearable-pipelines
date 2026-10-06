"""Compatibility import for the implementation in dare_wearables.lower_back.io.mcroberts_loader."""
import sys
from dare_wearables.lower_back.io import mcroberts_loader as _implementation

# Both historical paths resolve to the same module and function globals.
sys.modules[__name__] = _implementation
