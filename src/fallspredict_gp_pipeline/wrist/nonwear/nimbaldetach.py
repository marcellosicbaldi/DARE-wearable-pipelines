"""Compatibility import for the implementation in dare_wearables.wrist.nonwear.nimbaldetach."""
import sys
from dare_wearables.wrist.nonwear import nimbaldetach as _implementation

# Both historical paths resolve to the same module and function globals.
sys.modules[__name__] = _implementation
