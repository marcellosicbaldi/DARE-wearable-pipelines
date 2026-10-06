"""Compatibility import for the implementation in dare_wearables.wrist.circadian.activity_intensity."""
import sys
from dare_wearables.wrist.circadian import activity_intensity as _implementation

# Both historical paths resolve to the same module and function globals.
sys.modules[__name__] = _implementation
