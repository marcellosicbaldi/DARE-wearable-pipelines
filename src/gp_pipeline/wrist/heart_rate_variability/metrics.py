"""Compatibility import for the implementation in dare_wearables.wrist.heart_rate_variability.metrics."""
import sys
from dare_wearables.wrist.heart_rate_variability import metrics as _implementation

# Both historical paths resolve to the same module and function globals.
sys.modules[__name__] = _implementation
