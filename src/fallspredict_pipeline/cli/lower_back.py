from __future__ import annotations

import argparse

from ..config import LowerBackConfig
from ..lower_back.pipeline import run_from_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the DARE-FALLSPREDICT lower-back IMU pipeline.")
    parser.add_argument(
        "--config",
        default="configs/ravenna_lower_back.local.toml",
        help="Path to the TOML configuration file.",
    )
    parser.add_argument(
        "--participant",
        action="append",
        dest="participants",
        help="Optional participant override. Repeat to run multiple participants.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = LowerBackConfig.from_toml(args.config)
    if args.participants:
        config.participant_ids = [str(participant) for participant in args.participants]
    run_from_config(config)


if __name__ == "__main__":
    main()
