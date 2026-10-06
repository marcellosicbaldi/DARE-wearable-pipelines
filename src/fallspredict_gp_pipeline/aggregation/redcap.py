"""Compatibility import for the implementation in dare_wearables.aggregation.redcap."""
import sys
from dare_wearables.aggregation import redcap as _implementation

# Both historical paths resolve to the same module and function globals.
sys.modules[__name__] = _implementation

if __name__ == "__main__":
    _implementation.main()
