"""Compatibility import for the implementation in dare_wearables.wrist.heart_rate.io."""
import sys
from dare_wearables.wrist.heart_rate import io as _implementation

# Both historical paths resolve to the same module and function globals.
sys.modules[__name__] = _implementation
