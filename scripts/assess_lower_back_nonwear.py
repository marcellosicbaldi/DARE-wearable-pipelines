"""Nonwear-only cohort assessment; run from the repository's Python environment."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dare_wearables.lower_back.nonwear.assessment import run_assessment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--cohort", choices=("gp", "ravenna"), default="gp")
    parser.add_argument("--resume")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.cohort == "gp":
        from fallspredict_gp_pipeline.config import LowerBackConfig
    else:
        from fallspredict_pipeline.config import LowerBackConfig
    run_assessment(LowerBackConfig.from_toml(args.config), resume=args.resume,
                   inventory_only=args.inventory_only, workers=args.workers)


if __name__ == "__main__":
    main()
