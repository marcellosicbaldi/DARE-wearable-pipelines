"""Compatibility import for the implementation in dare_wearables.aggregation.analysis_exclusions."""
import sys
from dare_wearables.aggregation import analysis_exclusions as _implementation

# Both historical paths resolve to the same module and function globals.
sys.modules[__name__] = _implementation
