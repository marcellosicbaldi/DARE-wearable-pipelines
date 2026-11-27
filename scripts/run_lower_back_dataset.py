"""Prepare or execute the full GP lower-back pipeline in a separate run folder."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from dare_wearables.lower_back.batch import execute, prepare
from fallspredict_gp_pipeline.config import LowerBackConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path, help="Create a new run from this GP TOML configuration.")
    source.add_argument("--resume", type=Path, help="Execute/retry an existing frozen run folder.")
    parser.add_argument("--inventory-only", action="store_true", help="Freeze the inventory without reading sensor samples or running algorithms.")
    parser.add_argument("--participant", action="append", help="Optional explicit participants for a new run; omit for the whole dataset.")
    args = parser.parse_args()
    if args.resume and (args.inventory_only or args.participant):
        parser.error("Resume uses the already frozen inventory; do not supply inventory-only or participants.")
    if args.config:
        config = LowerBackConfig.from_toml(args.config)
        if args.participant:
            config.participant_ids = args.participant
        run = prepare(config)
        if args.inventory_only:
            return 0
    else:
        run = args.resume
    return 0 if execute(run) else 1


if __name__ == "__main__":
    raise SystemExit(main())
