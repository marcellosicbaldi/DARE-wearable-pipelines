"""Compatibility import for the implementation in dare_wearables.lower_back.utils.bool_to_bouts."""
import sys
from dare_wearables.lower_back.utils import bool_to_bouts as _implementation

# Both historical paths resolve to the same module and function globals.
sys.modules[__name__] = _implementation
