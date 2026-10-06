"""Compatibility import for the implementation in dare_wearables.wrist.circadian.empatica."""
import sys
from dare_wearables.wrist.circadian import empatica as _implementation

# Both historical paths resolve to the same module and function globals.
sys.modules[__name__] = _implementation
