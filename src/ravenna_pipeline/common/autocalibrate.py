"""Compatibility import for the implementation in dare_wearables.common.autocalibrate."""
import sys
from dare_wearables.common import autocalibrate as _implementation

# Both historical paths resolve to the same module and function globals.
sys.modules[__name__] = _implementation
