"""Compatibility import for the implementation in dare_wearables.lower_back.nonwear.vanhees2013."""
import sys
from dare_wearables.lower_back.nonwear import vanhees2013 as _implementation

# Both historical paths resolve to the same module and function globals.
sys.modules[__name__] = _implementation
