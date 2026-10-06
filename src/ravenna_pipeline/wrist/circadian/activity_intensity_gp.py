"""Compatibility import for the implementation in dare_wearables.wrist.circadian.activity_intensity_empatica."""
import sys
from dare_wearables.wrist.circadian import activity_intensity_empatica as _implementation

# Both historical paths resolve to the same module and function globals.
sys.modules[__name__] = _implementation
