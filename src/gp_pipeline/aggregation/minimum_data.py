"""Compatibility import for the implementation in dare_wearables.aggregation.minimum_data."""
import sys
from dare_wearables.aggregation import minimum_data as _implementation

# Both historical paths resolve to the same module and function globals.
sys.modules[__name__] = _implementation
