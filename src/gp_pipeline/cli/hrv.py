from __future__ import annotations

import argparse

from ..wrist.heart_rate_variability import HRVConfig, run_hrv_batch_from_config, run_hrv_pipeline_from_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the GP Empatica wrist HRV pipeline.")
    parser.add_argument(
        "--config",
        default="configs/hrv.local.toml",
        help="Path to the TOML configuration file.",
    )
    parser.add_argument(
        "--participant",
        action="append",
        dest="participants",
        help="Optional participant override. Repeat to run multiple participants.",
    )
    parser.add_argument("--visit", help="Visit to process, for example T0 or T1.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every participant folder found under input_root.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = HRVConfig.from_toml(args.config)

    if args.all or (args.participants and len(args.participants) > 1):
        results = run_hrv_batch_from_config(config, participants=args.participants, visit=args.visit)
        completed = sum(result["status"] == "completed" for result in results)
        skipped = sum(result["status"] == "skipped" for result in results)
        print(f"HRV batch completed. Completed: {completed}; skipped: {skipped}.")
        return

    participant = args.participants[0] if args.participants else config.participant
    if participant is None:
        raise ValueError("Please provide a participant via `--participant`, `--all`, or the config file.")

    result = run_hrv_pipeline_from_config(config, participant=participant, visit=args.visit)
    print(f"HRV pipeline {result['status']}. Outputs: {result.get('output_dir')}")


if __name__ == "__main__":
    main()
