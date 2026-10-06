"""Compatibility import for the implementation in dare_wearables.wrist.data_io.geneactiv."""
import sys
from dare_wearables.wrist.data_io import geneactiv as _implementation

# Both historical paths resolve to the same module and function globals.
sys.modules[__name__] = _implementation
