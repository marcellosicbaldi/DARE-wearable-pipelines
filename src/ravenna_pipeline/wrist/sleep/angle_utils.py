"""Compatibility import for the implementation in dare_wearables.wrist.sleep.angle_utils."""
import sys
from dare_wearables.wrist.sleep import angle_utils as _implementation

# Both historical paths resolve to the same module and function globals.
sys.modules[__name__] = _implementation
