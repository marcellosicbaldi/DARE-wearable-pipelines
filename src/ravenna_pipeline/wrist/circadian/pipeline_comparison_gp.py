"""Compatibility import for the implementation in dare_wearables.wrist.circadian.comparison_empatica."""
import sys
from dare_wearables.wrist.circadian import comparison_empatica as _implementation

# Both historical paths resolve to the same module and function globals.
sys.modules[__name__] = _implementation
