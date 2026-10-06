"""Compatibility import for the implementation in dare_wearables.wrist.heart_rate_variability.ppg_beat_detection."""
import sys
from dare_wearables.wrist.heart_rate_variability import ppg_beat_detection as _implementation

# Both historical paths resolve to the same module and function globals.
sys.modules[__name__] = _implementation
