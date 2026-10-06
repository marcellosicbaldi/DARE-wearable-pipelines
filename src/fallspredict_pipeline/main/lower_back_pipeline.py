from __future__ import annotations

import argparse
import importlib.util
import sys
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

DEFAULT_LOG_FILE = PROJECT_ROOT / "logs" / "ravenna_lower_back_pipeline_errors.txt"


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


REMOVED_WHEEL_PATHS = _remove_wheel_files_from_path()


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


def _discover_participants(bronze_root: Path) -> list[str]:
    if not bronze_root.exists() or not bronze_root.is_dir():
        return []
    return sorted(path.name for path in bronze_root.iterdir() if path.is_dir())


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
        handle.write(_import_diagnostics())
        handle.write("\n\n")
        handle.write(traceback.format_exc())
        handle.write("\n")


def _import_diagnostics() -> str:
    mobgap_spec = importlib.util.find_spec("mobgap")
    numpy_spec = importlib.util.find_spec("numpy")
    lines = [
        "Import diagnostics:",
        f"  Python executable: {sys.executable}",
        f"  Python version: {sys.version}",
        f"  Removed .whl sys.path entries: {REMOVED_WHEEL_PATHS or 'none'}",
        f"  mobgap spec: {mobgap_spec.origin if mobgap_spec else 'NOT FOUND'}",
        f"  numpy spec: {numpy_spec.origin if numpy_spec else 'NOT FOUND'}",
        "  sys.path:",
    ]
    lines.extend(f"    - {path}" for path in sys.path)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DARE-FALLSPREDICT lower-back pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "ravenna_lower_back.local.toml",
        help="Path to configs/ravenna_lower_back.local.toml.",
    )
    parser.add_argument(
        "--participant",
        action="append",
        dest="participants",
        help="Participant identifier. Repeat to run multiple participants. Defaults to folders in bronze_root.",
    )
    parser.add_argument(
        "--visit",
        action="append",
        dest="visits",
        help="Visit override. Repeat for multiple visits.",
    )
    parser.add_argument("--bronze-root", type=Path, help="Override lower_back.bronze_root.")
    parser.add_argument("--silver-root", type=Path, help="Override lower_back.silver_root.")
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
    parser.add_argument("--skip-gait", action="store_true", help="Do not run gait outputs.")
    parser.add_argument(
        "--skip-time-in-bed",
        action="store_true",
        help="Do not run posture/time-in-bed outputs.",
    )
    parser.add_argument(
        "--save-omx-to-parquet",
        action="store_true",
        help="Save the preprocessed OMX parquet file, overriding the TOML setting.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the loaded config before running.",
    )
    parser.add_argument(
        "--diagnose-imports",
        action="store_true",
        help="Print Python executable, numpy location, mobgap location, and sys.path, then exit.",
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

    if args.diagnose_imports:
        print(_import_diagnostics())
        return

    from fallspredict_pipeline.config import LowerBackConfig
    from fallspredict_pipeline.lower_back.pipeline import run_from_config

    config = LowerBackConfig.from_toml(args.config)
    if args.bronze_root is not None:
        config.bronze_root = args.bronze_root.expanduser().resolve()
    if args.silver_root is not None:
        config.silver_root = args.silver_root.expanduser().resolve()
    if args.visits:
        config.visits = [str(visit) for visit in args.visits]
    if args.skip_gait:
        config.run_gait = False
    if args.skip_time_in_bed:
        config.run_time_in_bed = False
    if args.save_omx_to_parquet:
        config.save_omx_to_parquet = True

    if args.participants:
        participants = [str(participant) for participant in args.participants]
    else:
        participants = _discover_participants(config.bronze_root)
        participants = _slice_participants(
            participants,
            start_index=args.start_index,
            limit=args.limit,
        )
        if not participants and config.participant_ids:
            participants = [str(participant) for participant in config.participant_ids]

    if not participants:
        raise ValueError(
            "No participants found. Pass --participant or check lower_back.bronze_root."
        )

    config.participant_ids = participants

    if args.print_config:
        print(config)

    print(f"Project root: {PROJECT_ROOT}")
    if wheelhouse is not None:
        print(f"Using wheelhouse: {wheelhouse}")
    print(f"Found {len(participants)} participant(s): {participants}")
    print(f"Errors will be logged to: {log_file}")

    for participant in participants:
        try:
            print(f"\nProcessing lower-back participant {participant}...")
            config.participant_ids = [participant]
            run_from_config(config)
            print(f"Completed lower-back participant {participant}.")
        except Exception as exc:
            visits = ",".join(config.visits)
            _log_exception(
                log_file,
                f"lower-back participant={participant} visits={visits}",
                exc,
            )
            print(f"ERROR for lower-back participant {participant}. Logged to: {log_file}")
            continue


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log_file = _get_log_file_from_argv(DEFAULT_LOG_FILE)
        _log_exception(log_file, "fatal lower-back pipeline error", exc)
        print(f"FATAL ERROR. Logged to: {log_file}")
        raise
