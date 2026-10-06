"""Compatibility import for the implementation in dare_wearables.common.output_state."""
import sys
from dare_wearables.common import output_state as _implementation

# Both historical paths resolve to the same module and function globals.
sys.modules[__name__] = _implementation
