from __future__ import annotations

import argparse

from ..config import RavennaSleepConfig
from ..wrist.pipeline import (
    build_output_dir,
    run_sleep_and_circadian_from_config,
    run_sleep_pipeline_from_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the DARE-FALLSPREDICT GENEActiv wrist sleep pipeline.")
    parser.add_argument(
        "--config",
        default="configs/ravenna_sleep.local.toml",
        help="Path to the DARE-FALLSPREDICT TOML configuration file.",
    )
    parser.add_argument("--participant", help="Participant identifier to process.")
    parser.add_argument("--visit", help="Visit to process, for example T0 or T1.")
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Also run the circadian and activity-intensity stages after sleep preprocessing.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = RavennaSleepConfig.from_toml(args.config)
    participant = args.participant or config.participant
    visit = args.visit or config.visit
    if participant is None:
        raise ValueError("Please provide a participant via `--participant` or the config file.")

    if args.combined:
        run_sleep_and_circadian_from_config(config, participant=participant, visit=visit)
    else:
        run_sleep_pipeline_from_config(config, participant=participant, visit=visit)

    output_dir = build_output_dir(
        config.output_root,
        participant=participant,
        visit=visit,
        sensor=config.sensor,
    )
    print(f"DARE-FALLSPREDICT wrist pipeline completed. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
