from __future__ import annotations

import argparse

from ..wrist.heart_rate import (
    HeartRateConfig,
    run_heart_rate_batch_from_config,
    run_heart_rate_from_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the GP Empatica wrist BeliefPPG heart-rate pipeline.")
    parser.add_argument(
        "--config",
        default="configs/heart_rate.local.toml",
        help="Path to the TOML configuration file.",
    )
    parser.add_argument(
        "--participant",
        action="append",
        dest="participants",
        help="Optional participant override. Repeat to run multiple participants.",
    )
    parser.add_argument(
        "--visit",
        action="append",
        dest="visits",
        help="Optional visit override. Repeat to run multiple visits.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every participant folder found under input_root.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = HeartRateConfig.from_toml(args.config)

    selected_visits = args.visits or config.visits
    selected_participants = args.participants
    if selected_participants is None and config.participant is not None and not args.all:
        selected_participants = [config.participant]

    if args.all or (selected_participants and len(selected_participants) > 1) or len(selected_visits) > 1:
        results = run_heart_rate_batch_from_config(
            config,
            participants=selected_participants,
            visits=selected_visits,
        )
        completed = sum(result["status"] == "completed" for result in results)
        skipped = sum(result["status"] == "skipped" for result in results)
        print(f"Heart-rate batch completed. Completed: {completed}; skipped: {skipped}.")
        return

    participant = selected_participants[0] if selected_participants else None
    visit = selected_visits[0] if selected_visits else None
    if participant is None:
        raise ValueError("Please provide a participant via `--participant`, `--all`, or the config file.")

    result = run_heart_rate_from_config(config, participant=participant, visit=visit)
    print(f"Heart-rate pipeline {result['status']}. Output: {result.get('output_path')}")


if __name__ == "__main__":
    main()
