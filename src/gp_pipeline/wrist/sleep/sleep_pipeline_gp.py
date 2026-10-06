"""Compatibility import for the implementation in dare_wearables.wrist.sleep.pipeline."""
import sys
from dare_wearables.wrist.sleep import pipeline as _implementation

# Both historical paths resolve to the same module and function globals.
sys.modules[__name__] = _implementation
