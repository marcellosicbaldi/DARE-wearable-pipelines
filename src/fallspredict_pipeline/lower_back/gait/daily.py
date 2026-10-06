"""Compatibility import for the implementation in dare_wearables.lower_back.gait.daily."""
import sys
from dare_wearables.lower_back.gait import daily as _implementation

# Both historical paths resolve to the same module and function globals.
sys.modules[__name__] = _implementation
