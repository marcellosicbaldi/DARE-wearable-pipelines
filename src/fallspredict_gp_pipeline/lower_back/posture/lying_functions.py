"""Compatibility import for the implementation in dare_wearables.lower_back.posture.lying_functions."""
import sys
from dare_wearables.lower_back.posture import lying_functions as _implementation

# Both historical paths resolve to the same module and function globals.
sys.modules[__name__] = _implementation
