"""Compatibility import for the implementation in dare_wearables.lower_back.utils.fill_gaps_with_nan."""
import sys
from dare_wearables.lower_back.utils import fill_gaps_with_nan as _implementation

# Both historical paths resolve to the same module and function globals.
sys.modules[__name__] = _implementation
