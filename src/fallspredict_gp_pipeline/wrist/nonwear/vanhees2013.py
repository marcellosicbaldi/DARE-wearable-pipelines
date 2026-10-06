"""Compatibility import for the implementation in dare_wearables.wrist.nonwear.vanhees2013."""
import sys
from dare_wearables.wrist.nonwear import vanhees2013 as _implementation

# Both historical paths resolve to the same module and function globals.
sys.modules[__name__] = _implementation
