from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from dare_wearables.aggregation.minimum_data import HR_COUNT_COLUMNS, apply_minimum_observations


DEFAULT_SILVER_ROOT = Path("~/dare-data/silver").expanduser()
DEFAULT_RECRUITMENT_TRACKER = Path(
    "~/dare-data/recruitment_tracker/recruitment_tracker.xlsx"
).expanduser()
DEFAULT_VISIT = "T0"
DEFAULT_HR_SUBDIR = Path("Empatica") / "beliefppg"
DEFAULT_HR_FILENAME = "hr_belief.csv"
DEFAULT_SLEEP_WINDOWS_SUBDIR = Path("Empatica") / "hrv"
DEFAULT_SLEEP_WINDOWS_FILENAME = "hrv_sleep_windows_selected.csv"
DEFAULT_TIME_SHIFT_SECONDS = 2.0


def _discover_heart_rate_inputs(
    silver_root: Path,
    *,
    visit: str,
    hr_subdir: Path,
    hr_filename: str,
    sleep_windows_subdir: Path,
    sleep_windows_filename: str,
) -> list[tuple[str, Path, Path]]:
    hr_pattern = str(Path("*") / visit / hr_subdir / hr_filename)
    sleep_pattern = str(Path("*") / visit / sleep_windows_subdir / sleep_windows_filename)
    hr_files = {path.parents[3].name: path for path in silver_root.glob(hr_pattern)}
    sleep_window_files = {path.parents[3].name: path for path in silver_root.glob(sleep_pattern)}

    subjects = sorted(set(hr_files).intersection(sleep_window_files))
    return [(subject, hr_files[subject], sleep_window_files[subject]) for subject in subjects]


def _read_hr_file(path: Path, *, time_shift_seconds: float = DEFAULT_TIME_SHIFT_SECONDS) -> pd.DataFrame:
    hr_df = pd.read_csv(path, index_col=0)
    hr_df.index = pd.to_datetime(hr_df.index, errors="coerce")
    hr_df = hr_df.loc[hr_df.index.notna()].copy()
    hr_df.index = hr_df.index + pd.Timedelta(seconds=float(time_shift_seconds))
    if "hr" not in hr_df.columns:
        raise ValueError(f"Heart-rate file is missing required column 'hr': {path}")
    hr_df["hr"] = pd.to_numeric(hr_df["hr"], errors="coerce")
    return hr_df.sort_index()


def _read_sleep_windows(path: Path) -> pd.DataFrame:
    sleep_windows = pd.read_csv(path)
    required_columns = {"night_id", "spt_start", "spt_end"}
    missing = required_columns.difference(sleep_windows.columns)
    if missing:
        raise ValueError(f"Sleep-window file is missing required columns {sorted(missing)}: {path}")

    sleep_windows = sleep_windows.copy()
    sleep_windows["spt_start"] = pd.to_datetime(sleep_windows["spt_start"], errors="coerce")
    sleep_windows["spt_end"] = pd.to_datetime(sleep_windows["spt_end"], errors="coerce")
    sleep_windows = sleep_windows.dropna(subset=["spt_start", "spt_end"])
    sleep_windows = sleep_windows.loc[sleep_windows["spt_end"] > sleep_windows["spt_start"]]
    return sleep_windows.sort_values("spt_start").reset_index(drop=True)


def _read_recruitment_tracker(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    path = Path(path).expanduser()
    if not path.exists():
        return pd.DataFrame()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Data Validation extension is not supported and will be removed",
            category=UserWarning,
            module="openpyxl",
        )
        return pd.read_excel(path, sheet_name="Recruitment", skiprows=1)


def _participant_as_int(subject: str) -> int | None:
    try:
        return int(str(subject))
    except ValueError:
        return None


def _start_date_from_tracker(
    recruitment_tracker: pd.DataFrame,
    *,
    subject: str,
    visit: str,
) -> pd.Timestamp | None:
    if recruitment_tracker.empty or "codice pseudo partecipante" not in recruitment_tracker.columns:
        return None

    subject_int = _participant_as_int(subject)
    if subject_int is None:
        return None

    date_column = f"Data {visit} reale"
    if date_column not in recruitment_tracker.columns:
        return None

    participant_ids = pd.to_numeric(
        recruitment_tracker["codice pseudo partecipante"],
        errors="coerce",
    )
    match = recruitment_tracker.loc[participant_ids == subject_int, date_column]
    if match.empty:
        return None

    start_date = pd.to_datetime(match.iloc[0], errors="coerce", dayfirst=True)
    if pd.isna(start_date):
        return None
    return start_date


def _trim_to_tracker_window(
    hr_df: pd.DataFrame,
    *,
    start_date: pd.Timestamp | None,
    days: int = 7,
) -> pd.DataFrame:
    if start_date is None:
        return hr_df
    end_date = pd.to_datetime(start_date) + pd.Timedelta(days=days)
    return hr_df.loc[(hr_df.index >= pd.to_datetime(start_date)) & (hr_df.index < end_date)].copy()


def _filter_extreme_hr_values(values: pd.Series) -> pd.Series:
    hr_values = pd.to_numeric(values, errors="coerce")
    hr_values = hr_values.mask(hr_values <= 0)
    valid_values = hr_values.dropna()
    if valid_values.empty:
        return hr_values

    median = valid_values.median()
    iqr = valid_values.quantile(0.75) - valid_values.quantile(0.25)
    threshold_lower = median - 3 * iqr
    threshold_upper = median + 3 * iqr
    outlier_mask = (
        hr_values.notna()
        & ((hr_values < threshold_lower) | (hr_values > threshold_upper))
    )
    return hr_values.mask(outlier_mask)


def _filtered_hr_median(values: pd.Series) -> float:
    filtered = _filter_extreme_hr_values(values)
    if filtered.dropna().empty:
        return np.nan
    return float(filtered.median())


def aggregate_participant_heart_rate(
    subject: str,
    visit: str,
    hr_file: Path,
    sleep_windows_file: Path,
    *,
    recruitment_tracker: pd.DataFrame,
    time_shift_seconds: float = DEFAULT_TIME_SHIFT_SECONDS,
) -> pd.DataFrame:
    hr_df = _read_hr_file(hr_file, time_shift_seconds=time_shift_seconds)
    start_date = _start_date_from_tracker(
        recruitment_tracker,
        subject=subject,
        visit=visit,
    )
    hr_df = _trim_to_tracker_window(hr_df, start_date=start_date)
    sleep_windows = _read_sleep_windows(sleep_windows_file)

    rows: list[dict[str, object]] = []
    for i, row in sleep_windows.iterrows():
        spt_start = row["spt_start"]
        spt_end = row["spt_end"]
        night_mask = (hr_df.index >= spt_start) & (hr_df.index <= spt_end)
        median_hr_night = _filtered_hr_median(hr_df.loc[night_mask, "hr"])

        if i == 0:
            day_start = pd.NaT
            day_end = pd.NaT
            median_hr_day = np.nan
        else:
            day_start = sleep_windows.loc[i - 1, "spt_end"]
            day_end = spt_start
            day_mask = (hr_df.index > day_start) & (hr_df.index < day_end)
            median_hr_day = _filtered_hr_median(hr_df.loc[day_mask, "hr"])

        if pd.notna(median_hr_day) and median_hr_day > 0 and pd.notna(median_hr_night):
            hr_dip_pct = ((median_hr_day - median_hr_night) / median_hr_day) * 100
        else:
            hr_dip_pct = np.nan

        rows.append(
            {
                "subject": str(subject),
                "visit": str(visit),
                "night_id": row["night_id"],
                "day_start": day_start,
                "day_end": day_end,
                "spt_start": spt_start,
                "spt_end": spt_end,
                "median_hr_day": median_hr_day,
                "median_hr_night": median_hr_night,
                "hr_dip_pct": hr_dip_pct,
            }
        )

    return pd.DataFrame(rows)


def aggregate_heart_rate(
    silver_root: str | Path = DEFAULT_SILVER_ROOT,
    *,
    recruitment_tracker_path: str | Path | None = DEFAULT_RECRUITMENT_TRACKER,
    visit: str = DEFAULT_VISIT,
    hr_subdir: str | Path = DEFAULT_HR_SUBDIR,
    hr_filename: str = DEFAULT_HR_FILENAME,
    sleep_windows_subdir: str | Path = DEFAULT_SLEEP_WINDOWS_SUBDIR,
    sleep_windows_filename: str = DEFAULT_SLEEP_WINDOWS_FILENAME,
    time_shift_seconds: float = DEFAULT_TIME_SHIFT_SECONDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    silver_root = Path(silver_root).expanduser()
    recruitment_tracker = _read_recruitment_tracker(recruitment_tracker_path)

    nightly_frames: list[pd.DataFrame] = []
    for subject, hr_file, sleep_windows_file in _discover_heart_rate_inputs(
        silver_root,
        visit=visit,
        hr_subdir=Path(hr_subdir),
        hr_filename=hr_filename,
        sleep_windows_subdir=Path(sleep_windows_subdir),
        sleep_windows_filename=sleep_windows_filename,
    ):
        nightly_df = aggregate_participant_heart_rate(
            subject,
            visit,
            hr_file,
            sleep_windows_file,
            recruitment_tracker=recruitment_tracker,
            time_shift_seconds=time_shift_seconds,
        )
        if not nightly_df.empty:
            nightly_frames.append(nightly_df)

    nightly_all = pd.concat(nightly_frames, ignore_index=True) if nightly_frames else pd.DataFrame()
    if nightly_all.empty:
        subject_level = pd.DataFrame(
            columns=["subject", "visit", *HR_COUNT_COLUMNS, *HR_COUNT_COLUMNS.values()]
        )
        return subject_level, nightly_all

    nightly_median = (
        nightly_all.groupby(["subject", "night_id"], sort=True)
        .median(numeric_only=True)
        .reset_index()
    )
    subject_level = (
        nightly_median.groupby("subject", sort=True)[
            ["median_hr_day", "median_hr_night", "hr_dip_pct"]
        ]
        .median()
        .reset_index()
    )
    subject_level.insert(1, "visit", str(visit))
    # Count only nonmissing estimates, once per night_id. Daytime HR can have
    # fewer observations than nocturnal HR; the first sleep window has no
    # preceding daytime estimate. Dipping needs paired day and night data.
    counts = nightly_median.groupby("subject")[list(HR_COUNT_COLUMNS)].count()
    for metric, count_column in HR_COUNT_COLUMNS.items():
        subject_level[count_column] = subject_level["subject"].map(counts[metric]).astype(int)
    return apply_minimum_observations(subject_level), nightly_all


def write_heart_rate_exports(
    silver_root: str | Path = DEFAULT_SILVER_ROOT,
    *,
    recruitment_tracker_path: str | Path | None = DEFAULT_RECRUITMENT_TRACKER,
    visit: str = DEFAULT_VISIT,
    output_dir: str | Path | None = None,
    hr_subdir: str | Path = DEFAULT_HR_SUBDIR,
    hr_filename: str = DEFAULT_HR_FILENAME,
    sleep_windows_subdir: str | Path = DEFAULT_SLEEP_WINDOWS_SUBDIR,
    sleep_windows_filename: str = DEFAULT_SLEEP_WINDOWS_FILENAME,
    time_shift_seconds: float = DEFAULT_TIME_SHIFT_SECONDS,
) -> tuple[Path, pd.DataFrame]:
    silver_root = Path(silver_root).expanduser()
    output_dir = Path(output_dir).expanduser() if output_dir is not None else silver_root / "aggregation"
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_df, _ = aggregate_heart_rate(
        silver_root,
        recruitment_tracker_path=recruitment_tracker_path,
        visit=visit,
        hr_subdir=hr_subdir,
        hr_filename=hr_filename,
        sleep_windows_subdir=sleep_windows_subdir,
        sleep_windows_filename=sleep_windows_filename,
        time_shift_seconds=time_shift_seconds,
    )
    output_path = output_dir / f"beliefppg_hr_{visit}_median.csv"
    subject_df.to_csv(output_path, index=False)
    return output_path, subject_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate T0 BeliefPPG heart-rate silver-layer outputs.")
    parser.add_argument(
        "--silver-root",
        default=str(DEFAULT_SILVER_ROOT),
        help="Root folder containing silver/<subject>/<visit>/Empatica/beliefppg outputs.",
    )
    parser.add_argument(
        "--recruitment-tracker",
        default=str(DEFAULT_RECRUITMENT_TRACKER),
        help="Recruitment tracker Excel file used to trim each recording to the visit's 7-day window.",
    )
    parser.add_argument(
        "--visit",
        default=DEFAULT_VISIT,
        help="Visit to aggregate. Defaults to T0.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Folder for aggregate CSV exports. Defaults to <silver-root>/aggregation.",
    )
    parser.add_argument(
        "--hr-subdir",
        default=str(DEFAULT_HR_SUBDIR),
        help="Path under each subject/visit folder containing hr_belief.csv.",
    )
    parser.add_argument(
        "--sleep-windows-subdir",
        default=str(DEFAULT_SLEEP_WINDOWS_SUBDIR),
        help="Path under each subject/visit folder containing selected sleep windows.",
    )
    parser.add_argument(
        "--time-shift-seconds",
        type=float,
        default=DEFAULT_TIME_SHIFT_SECONDS,
        help="Seconds added to BeliefPPG timestamps before segmentation. Defaults to 2.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_path, subject_df = write_heart_rate_exports(
        args.silver_root,
        recruitment_tracker_path=args.recruitment_tracker,
        visit=args.visit,
        output_dir=args.output_dir,
        hr_subdir=args.hr_subdir,
        sleep_windows_subdir=args.sleep_windows_subdir,
        time_shift_seconds=args.time_shift_seconds,
    )
    print(f"Wrote BeliefPPG heart-rate aggregation: {output_path} ({len(subject_df)} subjects)")


if __name__ == "__main__":
    main()
