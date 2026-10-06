"""Compatibility import for the implementation in dare_wearables.wrist.sleep.vh2018_spt."""
import sys
from dare_wearables.wrist.sleep import vh2018_spt as _implementation

# Both historical paths resolve to the same module and function globals.
sys.modules[__name__] = _implementation
