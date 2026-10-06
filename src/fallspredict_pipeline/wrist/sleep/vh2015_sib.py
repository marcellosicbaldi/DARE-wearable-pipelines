"""Compatibility import for the implementation in dare_wearables.wrist.sleep.vh2015_sib."""
import sys
from dare_wearables.wrist.sleep import vh2015_sib as _implementation

# Both historical paths resolve to the same module and function globals.
sys.modules[__name__] = _implementation
