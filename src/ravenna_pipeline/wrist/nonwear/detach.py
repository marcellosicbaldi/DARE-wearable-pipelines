"""Compatibility import for the implementation in dare_wearables.wrist.nonwear.detach."""
import sys
from dare_wearables.wrist.nonwear import detach as _implementation

# Both historical paths resolve to the same module and function globals.
sys.modules[__name__] = _implementation
