from __future__ import annotations

import argparse
import sys
import warnings
import traceback
from datetime import datetime
from pathlib import Path


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src").is_dir():
            return candidate

    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src").is_dir():
            return candidate

    return here.parents[3]


PROJECT_ROOT = _find_project_root()
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

warnings.filterwarnings("ignore", category=FutureWarning, module="openpyxl")

DEFAULT_LOG_FILE = PROJECT_ROOT / "logs" / "ravenna_wrist_pipeline_errors.txt"


def _remove_wheel_files_from_path() -> list[str]:
    removed = []
    kept = []
    for entry in sys.path:
        if str(entry).lower().endswith(".whl"):
            removed.append(str(entry))
        else:
            kept.append(entry)
    if removed:
        sys.path[:] = kept
    return removed


_remove_wheel_files_from_path()


def _discover_wheelhouse() -> Path | None:
    candidates = [
        PROJECT_ROOT / "fallspredict_python_packages",
        PROJECT_ROOT.parent / "fallspredict_python_packages",
        PROJECT_ROOT.parent.parent / "fallspredict_python_packages",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _add_wheelhouse_to_path(wheelhouse: Path | None) -> None:
    if wheelhouse is None:
        return
    wheelhouse = wheelhouse.expanduser().resolve()
    if not wheelhouse.exists():
        return

    entries = [
        path
        for path in sorted(wheelhouse.rglob("site-packages"))
        if path.is_dir()
    ]
    entries.extend(
        path
        for path in sorted(wheelhouse.iterdir())
        if path.is_dir() and (path / "__init__.py").exists()
    )
    for entry in reversed(entries):
        text = str(entry)
        if text not in sys.path:
            sys.path.insert(0, text)
    _remove_wheel_files_from_path()


def _discover_participants(input_root: Path) -> list[str]:
    if not input_root.exists() or not input_root.is_dir():
        return []
    return sorted(path.name for path in input_root.iterdir() if path.is_dir())


def _slice_participants(
    participants: list[str],
    *,
    start_index: int = 0,
    limit: int | None = None,
) -> list[str]:
    selected = participants[max(start_index, 0):]
    if limit is not None:
        selected = selected[: max(limit, 0)]
    return selected


def _get_log_file_from_argv(default: Path) -> Path:
    for i, arg in enumerate(sys.argv):
        if arg == "--log-file" and i + 1 < len(sys.argv):
            return Path(sys.argv[i + 1]).expanduser().resolve()
        if arg.startswith("--log-file="):
            return Path(arg.split("=", 1)[1]).expanduser().resolve()
    return default


def _log_exception(log_file: Path, context: str, exc: BaseException) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write("\n" + "=" * 80 + "\n")
        handle.write(f"{datetime.now().isoformat(timespec='seconds')} | {context}\n")
        handle.write(f"{type(exc).__name__}: {exc}\n\n")
        handle.write(traceback.format_exc())
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DARE-FALLSPREDICT GENEActiv wrist sleep/circadian/activity pipeline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "ravenna_sleep.local.toml",
        help="Path to configs/ravenna_sleep.local.toml.",
    )
    parser.add_argument(
        "--participant",
        action="append",
        dest="participants",
        help="Participant identifier. Repeat to run multiple participants. Defaults to folders in input_root.",
    )
    parser.add_argument("--visit", help="Visit override, for example T0 or T1.")
    parser.add_argument(
        "--input-root",
        type=Path,
        help="Override the GENEActiv input root from the TOML.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Override the output root from the TOML.",
    )
    parser.add_argument(
        "--wheelhouse",
        type=Path,
        help="Folder containing offline .whl packages, for example DARE/codice/fallspredict_python_packages.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start at this index in the discovered participant list.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most this many participants after start-index.",
    )
    parser.add_argument(
        "--sleep-only",
        action="store_true",
        help="Run only the sleep pipeline. Default is sleep + circadian + activity-intensity.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip participants with sleep_output_all_guiders.csv already present.",
    )
    parser.add_argument(
        "--no-lower-back-tib",
        action="store_true",
        help="Do not attach the lower-back tib_valid_df.csv guider for each participant.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the loaded config before running.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_FILE,
        help="Text file where participant errors and fatal errors are appended.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    log_file = args.log_file.expanduser().resolve()
    wheelhouse = args.wheelhouse.expanduser().resolve() if args.wheelhouse else _discover_wheelhouse()
    _add_wheelhouse_to_path(wheelhouse)

    from fallspredict_pipeline.config import RavennaSleepConfig
    from fallspredict_pipeline.wrist.pipeline import (
        build_output_dir,
        run_sleep_and_circadian_from_config,
        run_sleep_pipeline_from_config,
    )

    config = RavennaSleepConfig.from_toml(args.config)
    if args.input_root is not None:
        config.input_root = args.input_root.expanduser().resolve()
    if args.output_root is not None:
        config.output_root = args.output_root.expanduser().resolve()
    if args.visit is not None:
        config.visit = args.visit

    if args.print_config:
        print(config)

    if args.participants:
        participants = [str(participant) for participant in args.participants]
    else:
        participants = _discover_participants(config.input_root)
        participants = _slice_participants(
            participants,
            start_index=args.start_index,
            limit=args.limit,
        )
        if not participants and config.participant:
            participants = [config.participant]

    if not participants:
        raise ValueError(
            "No participants found. Pass --participant or check ravenna_sleep.input_root."
        )

    visit = config.visit
    print(f"Project root: {PROJECT_ROOT}")
    if wheelhouse is not None:
        print(f"Using wheelhouse: {wheelhouse}")
    print(f"Found {len(participants)} participant(s): {participants}")
    print(f"Errors will be logged to: {log_file}")

    for participant in participants:
        try:
            print(f"\nProcessing wrist participant {participant}...")
            config.participant = participant

            if args.no_lower_back_tib:
                config.lb_tib_csv_path = None
            else:
                config.lb_tib_csv_path = (
                    Path(config.output_root)
                    / participant
                    / visit
                    / "McRoberts"
                    / "posture"
                    / "tib_valid_df.csv"
                )

            output_dir = build_output_dir(
                config.output_root,
                participant=participant,
                visit=visit,
                sensor=config.sensor,
            )
            if args.skip_existing and (output_dir / "sleep_output_all_guiders.csv").exists():
                print(f"Output already exists for {participant} {visit}. Skipping: {output_dir}")
                continue

            if args.sleep_only:
                run_sleep_pipeline_from_config(config, participant=participant, visit=visit)
            else:
                run_sleep_and_circadian_from_config(config, participant=participant, visit=visit)

            print(f"Completed wrist participant {participant}. Outputs: {output_dir}")
        except Exception as exc:
            _log_exception(log_file, f"wrist participant={participant} visit={visit}", exc)
            print(f"ERROR for wrist participant {participant}. Logged to: {log_file}")
            continue


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log_file = _get_log_file_from_argv(DEFAULT_LOG_FILE)
        _log_exception(log_file, "fatal wrist pipeline error", exc)
        print(f"FATAL ERROR. Logged to: {log_file}")
        raise
