"""Compatibility import for the implementation in dare_wearables.common.recording."""
import sys
from dare_wearables.common import recording as _implementation

# Both historical paths resolve to the same module and function globals.
sys.modules[__name__] = _implementation
