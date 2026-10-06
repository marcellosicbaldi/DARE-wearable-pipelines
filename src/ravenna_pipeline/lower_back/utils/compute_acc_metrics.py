"""Compatibility import for the implementation in dare_wearables.lower_back.utils.compute_acc_metrics."""
import sys
from dare_wearables.lower_back.utils import compute_acc_metrics as _implementation

# Both historical paths resolve to the same module and function globals.
sys.modules[__name__] = _implementation
