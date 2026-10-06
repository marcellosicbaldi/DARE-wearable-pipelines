from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dare_wearables.aggregation.minimum_data import HRV_METRICS, apply_minimum_observations


DEFAULT_SILVER_ROOT = Path("~/dare-data/silver").expanduser()
DEFAULT_VISIT = "T0"
DEFAULT_HRV_SUBDIR = Path("Empatica") / "hrv"
DEFAULT_HRV_FILENAME = "hrv_night.csv"

METRICS_TO_FILTER = ("rmssd", "mean_hr", "sdnn", "PIP")
METRIC_PREFIX = "hrv"


def _discover_hrv_files(
    silver_root: Path,
    *,
    visit: str,
    hrv_subdir: Path,
    hrv_filename: str,
) -> list[tuple[str, Path]]:
    pattern = str(Path("*") / visit / hrv_subdir / hrv_filename)
    hrv_files = sorted(silver_root.glob(pattern))
    return [(path.parents[3].name, path) for path in hrv_files]


def _read_hrv_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    if "day" not in df.columns:
        df = pd.read_csv(path)
    return df


def _empty_thresholds() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "subject",
            "metric",
            "threshold_lower",
            "threshold_upper",
            "threshold_median",
            "threshold_iqr",
        ]
    )


def _empty_discarded_by_night() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "subject",
            "day",
            "metric",
            "n_windows",
            "n_discarded",
            "percent_discarded",
            "threshold_lower",
            "threshold_upper",
            "threshold_median",
            "threshold_iqr",
        ]
    )


def filter_participant_hrv(
    hrv_df: pd.DataFrame,
    *,
    subject: str,
    metrics_to_filter: tuple[str, ...] = METRICS_TO_FILTER,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hrv_filtered = hrv_df.copy()
    hrv_filtered["subject"] = str(subject)

    available_metrics = [metric for metric in metrics_to_filter if metric in hrv_filtered.columns]
    if not available_metrics:
        return hrv_filtered.iloc[0:0].copy(), _empty_thresholds(), _empty_discarded_by_night()

    thresholds: list[dict[str, object]] = []

    for metric in available_metrics:
        metric_values = pd.to_numeric(hrv_filtered[metric], errors="coerce")
        metric_values = metric_values.mask(metric_values <= 0)
        hrv_filtered[metric] = metric_values

        valid_values = metric_values.dropna()
        outlier_col = f"{metric}_outlier"
        if valid_values.empty:
            hrv_filtered[outlier_col] = False
            continue

        metric_median = valid_values.median()
        metric_iqr = valid_values.quantile(0.75) - valid_values.quantile(0.25)
        threshold_upper = metric_median + 3 * metric_iqr
        threshold_lower = metric_median - 3 * metric_iqr

        outlier_mask = (
            metric_values.notna()
            & ((metric_values < threshold_lower) | (metric_values > threshold_upper))
        )
        hrv_filtered[outlier_col] = outlier_mask
        hrv_filtered.loc[outlier_mask, metric] = np.nan

        thresholds.append(
            {
                "subject": str(subject),
                "metric": metric,
                "threshold_lower": threshold_lower,
                "threshold_upper": threshold_upper,
                "threshold_median": metric_median,
                "threshold_iqr": metric_iqr,
            }
        )

    thresholds_df = pd.DataFrame(thresholds)
    discarded_by_night = _discarded_windows_by_night(
        hrv_filtered,
        subject=str(subject),
        available_metrics=available_metrics,
        thresholds_df=thresholds_df,
    )
    return hrv_filtered, thresholds_df, discarded_by_night


def _discarded_windows_by_night(
    hrv_filtered: pd.DataFrame,
    *,
    subject: str,
    available_metrics: list[str],
    thresholds_df: pd.DataFrame,
) -> pd.DataFrame:
    if "day" not in hrv_filtered.columns:
        return _empty_discarded_by_night()

    discarded_rows: list[pd.DataFrame] = []
    for metric in available_metrics:
        outlier_col = f"{metric}_outlier"
        if outlier_col not in hrv_filtered.columns:
            continue

        n_discarded_metric = (
            hrv_filtered.groupby("day")
            .agg(
                n_windows=(outlier_col, "size"),
                n_discarded=(outlier_col, "sum"),
            )
            .reset_index()
        )
        n_discarded_metric["percent_discarded"] = (
            n_discarded_metric["n_discarded"] / n_discarded_metric["n_windows"] * 100
        )
        n_discarded_metric["subject"] = str(subject)
        n_discarded_metric["metric"] = metric

        if not thresholds_df.empty:
            metric_thresholds = thresholds_df.loc[thresholds_df["metric"] == metric]
            if not metric_thresholds.empty:
                for threshold_col in (
                    "threshold_lower",
                    "threshold_upper",
                    "threshold_median",
                    "threshold_iqr",
                ):
                    n_discarded_metric[threshold_col] = metric_thresholds[threshold_col].iloc[0]

        for threshold_col in (
            "threshold_lower",
            "threshold_upper",
            "threshold_median",
            "threshold_iqr",
        ):
            if threshold_col not in n_discarded_metric.columns:
                n_discarded_metric[threshold_col] = np.nan

        discarded_rows.append(n_discarded_metric)

    if not discarded_rows:
        return _empty_discarded_by_night()

    out = pd.concat(discarded_rows, ignore_index=True)
    return out[
        [
            "subject",
            "day",
            "metric",
            "n_windows",
            "n_discarded",
            "percent_discarded",
            "threshold_lower",
            "threshold_upper",
            "threshold_median",
            "threshold_iqr",
        ]
    ]


def load_and_filter_hrv(
    silver_root: str | Path = DEFAULT_SILVER_ROOT,
    *,
    visit: str = DEFAULT_VISIT,
    hrv_subdir: str | Path = DEFAULT_HRV_SUBDIR,
    hrv_filename: str = DEFAULT_HRV_FILENAME,
    metrics_to_filter: tuple[str, ...] = METRICS_TO_FILTER,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    silver_root = Path(silver_root).expanduser()
    hrv_subdir = Path(hrv_subdir)

    raw_frames: list[pd.DataFrame] = []
    filtered_frames: list[pd.DataFrame] = []
    threshold_frames: list[pd.DataFrame] = []
    discarded_frames: list[pd.DataFrame] = []

    for subject, hrv_file in _discover_hrv_files(
        silver_root,
        visit=visit,
        hrv_subdir=hrv_subdir,
        hrv_filename=hrv_filename,
    ):
        hrv_df = _read_hrv_file(hrv_file)
        if hrv_df.empty:
            continue

        hrv_df = hrv_df.copy()
        hrv_df["subject"] = str(subject)
        hrv_filtered, thresholds_df, discarded_by_night = filter_participant_hrv(
            hrv_df,
            subject=subject,
            metrics_to_filter=metrics_to_filter,
        )
        if hrv_filtered.empty:
            continue

        raw_frames.append(hrv_df)
        filtered_frames.append(hrv_filtered)
        if not thresholds_df.empty:
            threshold_frames.append(thresholds_df)
        if not discarded_by_night.empty:
            discarded_frames.append(discarded_by_night)

    raw_df = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    filtered_df = pd.concat(filtered_frames, ignore_index=True) if filtered_frames else pd.DataFrame()
    thresholds_df = (
        pd.concat(threshold_frames, ignore_index=True) if threshold_frames else _empty_thresholds()
    )
    discarded_df = (
        pd.concat(discarded_frames, ignore_index=True)
        if discarded_frames
        else _empty_discarded_by_night()
    )
    return raw_df, filtered_df, thresholds_df, discarded_df


def aggregate_hrv_windows(filtered_df: pd.DataFrame, *, visit: str = DEFAULT_VISIT) -> pd.DataFrame:
    if filtered_df.empty or "subject" not in filtered_df.columns or "day" not in filtered_df.columns:
        return pd.DataFrame(columns=["subject", "visit", "hrv_n_valid_nights", "hrv_n_windows"])

    outlier_cols = [col for col in filtered_df.columns if col.endswith("_outlier")]
    hrv_for_aggregation = filtered_df.drop(columns=outlier_cols)

    per_night = (
        hrv_for_aggregation.groupby(["subject", "day"], sort=True)
        .median(numeric_only=True)
        .reset_index()
    )
    numeric_columns = [
        column
        for column in per_night.select_dtypes(include="number").columns
        if column != "day"
    ]

    subject_level = per_night.groupby("subject", sort=True)[numeric_columns].median().reset_index()
    subject_level.insert(1, "visit", str(visit))

    # A stored night with every physiological metric filtered out is not a
    # valid HRV night. Counts also differ between metrics after outlier removal.
    available_metrics = [metric for metric in HRV_METRICS if metric in per_night]
    has_metric = per_night[available_metrics].notna().any(axis=1)
    n_valid_nights = per_night.loc[has_metric].groupby("subject")["day"].nunique()
    n_windows = filtered_df.groupby("subject").size()
    subject_level.insert(
        2,
        "hrv_n_valid_nights",
        subject_level["subject"].map(n_valid_nights).fillna(0).astype(int),
    )
    subject_level.insert(
        3,
        "hrv_n_windows",
        subject_level["subject"].map(n_windows).fillna(0).astype(int),
    )

    rename_map = {
        column: f"{METRIC_PREFIX}_{column}"
        for column in numeric_columns
        if not column.startswith(f"{METRIC_PREFIX}_")
    }
    subject_level = subject_level.rename(columns=rename_map)
    metric_counts = per_night.groupby("subject")[available_metrics].count()
    for metric in available_metrics:
        subject_level[f"hrv_{metric}_n_valid_nights"] = (
            subject_level["subject"].map(metric_counts[metric]).fillna(0).astype(int)
        )
    return apply_minimum_observations(subject_level)


def aggregate_discarded_percent(
    discarded_df: pd.DataFrame,
    *,
    visit: str = DEFAULT_VISIT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if discarded_df.empty:
        columns = ["subject", "visit"] + [
            f"{METRIC_PREFIX}_{metric}_percent_discarded_median"
            for metric in METRICS_TO_FILTER
        ]
        return pd.DataFrame(columns=columns), pd.DataFrame(
            columns=["subject", "visit", "metric", "percent_discarded"]
        )

    long_df = (
        discarded_df.groupby(["subject", "day", "metric"])["percent_discarded"]
        .median()
        .reset_index()
        .groupby(["subject", "metric"])["percent_discarded"]
        .median()
        .reset_index()
    )
    long_df.insert(1, "visit", str(visit))

    wide_df = long_df.pivot(
        index=["subject", "visit"],
        columns="metric",
        values="percent_discarded",
    ).reset_index()
    wide_df.columns.name = None
    wide_df = wide_df.rename(
        columns={
            metric: f"{METRIC_PREFIX}_{metric}_percent_discarded_median"
            for metric in wide_df.columns
            if metric not in {"subject", "visit"}
        }
    )
    return wide_df, long_df


def write_hrv_exports(
    silver_root: str | Path = DEFAULT_SILVER_ROOT,
    *,
    visit: str = DEFAULT_VISIT,
    output_dir: str | Path | None = None,
    hrv_subdir: str | Path = DEFAULT_HRV_SUBDIR,
    hrv_filename: str = DEFAULT_HRV_FILENAME,
    metrics_to_filter: tuple[str, ...] = METRICS_TO_FILTER,
) -> dict[str, object]:
    silver_root = Path(silver_root).expanduser()
    output_dir = Path(output_dir).expanduser() if output_dir is not None else silver_root / "aggregation"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df, filtered_df, thresholds_df, discarded_df = load_and_filter_hrv(
        silver_root,
        visit=visit,
        hrv_subdir=hrv_subdir,
        hrv_filename=hrv_filename,
        metrics_to_filter=metrics_to_filter,
    )
    subject_df = aggregate_hrv_windows(filtered_df, visit=visit)
    discarded_wide_df, discarded_long_df = aggregate_discarded_percent(discarded_df, visit=visit)

    paths = {
        "subject_median": output_dir / f"hrv_{visit}_median.csv",
        "percent_discarded_median": output_dir / f"hrv_{visit}_percent_discarded_median.csv",
        "percent_discarded_median_long": output_dir / f"hrv_{visit}_percent_discarded_median_long.csv",
        "discarded_by_night": output_dir / f"hrv_{visit}_discarded_by_night.csv",
        "outlier_thresholds": output_dir / f"hrv_{visit}_outlier_thresholds.csv",
    }
    subject_df.to_csv(paths["subject_median"], index=False)
    discarded_wide_df.to_csv(paths["percent_discarded_median"], index=False)
    discarded_long_df.to_csv(paths["percent_discarded_median_long"], index=False)
    discarded_df.to_csv(paths["discarded_by_night"], index=False)
    thresholds_df.to_csv(paths["outlier_thresholds"], index=False)

    return {
        "paths": paths,
        "raw": raw_df,
        "filtered": filtered_df,
        "subject_median": subject_df,
        "percent_discarded_median": discarded_wide_df,
        "percent_discarded_median_long": discarded_long_df,
        "discarded_by_night": discarded_df,
        "outlier_thresholds": thresholds_df,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate T0 HRV silver-layer outputs.")
    parser.add_argument(
        "--silver-root",
        default=str(DEFAULT_SILVER_ROOT),
        help="Root folder containing silver/<subject>/<visit>/Empatica/hrv outputs.",
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
        "--hrv-subdir",
        default=str(DEFAULT_HRV_SUBDIR),
        help="Path under each subject/visit folder containing HRV outputs.",
    )
    parser.add_argument(
        "--hrv-filename",
        default=DEFAULT_HRV_FILENAME,
        help="HRV window-level CSV filename. Defaults to hrv_night.csv.",
    )
    parser.add_argument(
        "--metrics-to-filter",
        nargs="+",
        default=list(METRICS_TO_FILTER),
        help="Metrics to filter using participant-level median +/- 3*IQR thresholds.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = write_hrv_exports(
        args.silver_root,
        visit=args.visit,
        output_dir=args.output_dir,
        hrv_subdir=args.hrv_subdir,
        hrv_filename=args.hrv_filename,
        metrics_to_filter=tuple(args.metrics_to_filter),
    )
    paths: dict[str, Path] = result["paths"]  # type: ignore[assignment]
    subject_df: pd.DataFrame = result["subject_median"]  # type: ignore[assignment]
    filtered_df: pd.DataFrame = result["filtered"]  # type: ignore[assignment]
    print(f"Wrote HRV subject median aggregation: {paths['subject_median']} ({len(subject_df)} subjects)")
    print(f"Wrote HRV discarded percentage summary: {paths['percent_discarded_median']}")
    print(f"Wrote HRV discarded-by-night QC: {paths['discarded_by_night']} ({len(filtered_df)} windows processed)")


if __name__ == "__main__":
    main()
