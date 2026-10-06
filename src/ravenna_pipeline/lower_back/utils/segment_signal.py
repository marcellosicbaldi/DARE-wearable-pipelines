"""Compatibility import for the implementation in dare_wearables.lower_back.utils.segment_signal."""
import sys
from dare_wearables.lower_back.utils import segment_signal as _implementation

# Both historical paths resolve to the same module and function globals.
sys.modules[__name__] = _implementation
