"""Compatibility import for the implementation in dare_wearables.wrist.sleep.sleep_functions."""
import sys
from dare_wearables.wrist.sleep import sleep_functions as _implementation

# Both historical paths resolve to the same module and function globals.
sys.modules[__name__] = _implementation
