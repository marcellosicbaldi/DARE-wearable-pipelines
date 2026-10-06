from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from dare_wearables.wrist.circadian.activity_intensity import (
    find_wrist_accelerometer_file,
    load_sleep_windows,
    run_activity_intensity_pipeline,
)
from dare_wearables.wrist.circadian.metrics import (
    circadian_pipeline,
    compare_against_ggir,
    load_ggir_part5_intensity,
)


INTENSITY_METRICS = [
    "dur_full_total_IN_min",
    "dur_full_total_LIG_min",
    "dur_full_total_MOD_min",
    "dur_full_total_VIG_min",
    "dur_full_total_MVPA_min",
]


def _normalize_subject_token(value: object) -> str:
    text = str(value).strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not extract a numeric subject identifier from {value!r}.")
    return str(int(digits))


def _resolve_subject_directory(root: str | Path, subject: str) -> Path:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    exact = root / str(subject)
    if exact.is_dir():
        return exact

    subject_norm = _normalize_subject_token(subject)
    matches = [
        path for path in root.iterdir()
        if path.is_dir() and _normalize_subject_token(path.name) == subject_norm
    ]
    if not matches:
        raise FileNotFoundError(f"Could not find subject {subject!r} under {root}.")
    if len(matches) > 1:
        raise ValueError(
            f"Found multiple matching subject folders for {subject!r}: {[path.name for path in matches]}"
        )
    return matches[0]


def _resolve_visit_directory(subject_dir: str | Path, visit: str = "T0") -> Path:
    subject_dir = Path(subject_dir)
    matches = sorted(
        path for path in subject_dir.iterdir()
        if path.is_dir() and path.name.startswith(visit)
    )
    if not matches:
        raise FileNotFoundError(f"Could not find a visit folder starting with {visit!r} in {subject_dir}.")
    return matches[0]


def _find_single_rdata(folder: str | Path) -> Path:
    folder = Path(folder)
    files = sorted(folder.glob("*.RData"))
    if not files:
        raise FileNotFoundError(f"No `.RData` files found in {folder}.")
    if len(files) > 1:
        raise ValueError(f"Found multiple `.RData` files in {folder}: {[path.name for path in files]}")
    return files[0]


def find_ggir_metric_files(ggir_root: str | Path,
                           subject: str,
                           visit: str = "T0",
                           sensor_folder: str = "GeneActivPolso") -> Dict[str, Path]:
    """
    Locate the GGIR part-5 and part-6 RData files for one subject/visit.
    """
    subject_dir = _resolve_subject_directory(ggir_root, subject)
    visit_dir = _resolve_visit_directory(subject_dir, visit=visit)
    meta_dir = visit_dir / sensor_folder / "output_icareit" / "meta"

    ms5_path = _find_single_rdata(meta_dir / "ms5.out")
    ms6_path = _find_single_rdata(meta_dir / "ms6.out")

    return {
        "ms5_rdata_path": ms5_path,
        "ms6_rdata_path": ms6_path,
    }


def _build_circadian_rows(subject: str,
                          visit: str,
                          circadian_results: Dict[str, object],
                          ggir_part6_path: str | Path) -> pd.DataFrame:
    comparison_df = compare_against_ggir(
        metrics_df=circadian_results["metrics_df"],
        ggir_rdata_path=str(ggir_part6_path),
    ).copy()
    comparison_df["subject"] = str(subject)
    comparison_df["visit"] = str(visit)
    comparison_df["domain"] = "circadian"
    comparison_df["aggregation"] = "summary"
    comparison_df["analysis_date"] = pd.NaT
    comparison_df = comparison_df.rename(
        columns={
            "python": "python_value",
            "ggir": "ggir_value",
        }
    )
    return comparison_df[
        [
            "subject",
            "visit",
            "domain",
            "aggregation",
            "analysis_date",
            "metric",
            "python_value",
            "ggir_value",
            "difference",
            "abs_difference",
        ]
    ]


def _build_activity_rows(subject: str,
                         visit: str,
                         intensity_daily_df: pd.DataFrame,
                         ggir_part5_path: str | Path) -> pd.DataFrame:
    py_daily = intensity_daily_df.copy()
    py_daily["analysis_date"] = pd.to_datetime(py_daily["analysis_date"]).dt.date

    ggir_daily = load_ggir_part5_intensity(
        ms5_rdata_path=str(ggir_part5_path),
        window_type="MM",
        drop_incomplete_days=True,
    ).copy()

    merged = py_daily.merge(
        ggir_daily,
        on="analysis_date",
        how="inner",
        suffixes=("_python", "_ggir"),
    )

    rows: List[Dict[str, object]] = []
    for _, row in merged.iterrows():
        for metric in INTENSITY_METRICS:
            python_metric = f"{metric}_python"
            ggir_metric = f"{metric}_ggir"
            rows.append(
                {
                    "subject": str(subject),
                    "visit": str(visit),
                    "domain": "activity_intensity",
                    "aggregation": "day",
                    "analysis_date": pd.to_datetime(row["analysis_date"]),
                    "metric": metric,
                    "python_value": float(row[python_metric]),
                    "ggir_value": float(row[ggir_metric]),
                    "difference": float(row[python_metric] - row[ggir_metric]),
                    "abs_difference": float(abs(row[python_metric] - row[ggir_metric])),
                }
            )

    if rows:
        per_day_df = pd.DataFrame(rows)
        summary_df = (
            per_day_df.groupby("metric")
            .agg(
                python_value=("python_value", "mean"),
                ggir_value=("ggir_value", "mean"),
                difference=("difference", "mean"),
                abs_difference=("abs_difference", "mean"),
            )
            .reset_index()
        )
        summary_df["subject"] = str(subject)
        summary_df["visit"] = str(visit)
        summary_df["domain"] = "activity_intensity"
        summary_df["aggregation"] = "summary"
        summary_df["analysis_date"] = pd.NaT
        summary_df = summary_df[
            [
                "subject",
                "visit",
                "domain",
                "aggregation",
                "analysis_date",
                "metric",
                "python_value",
                "ggir_value",
                "difference",
                "abs_difference",
            ]
        ]
        per_day_df = per_day_df[
            [
                "subject",
                "visit",
                "domain",
                "aggregation",
                "analysis_date",
                "metric",
                "python_value",
                "ggir_value",
                "difference",
                "abs_difference",
            ]
        ]
        return pd.concat([summary_df, per_day_df], ignore_index=True)

    return pd.DataFrame(
        columns=[
            "subject",
            "visit",
            "domain",
            "aggregation",
            "analysis_date",
            "metric",
            "python_value",
            "ggir_value",
            "difference",
            "abs_difference",
        ]
    )


def build_pipeline_comparison_dataframe(subject: str,
                                        visit: str,
                                        circadian_results: Dict[str, object],
                                        activity_results: Dict[str, object],
                                        ggir_part6_path: str | Path,
                                        ggir_part5_path: str | Path) -> pd.DataFrame:
    """
    Create one tidy comparison dataframe combining:
    - Python circadian metrics vs GGIR part 6
    - Python activity-intensity metrics vs GGIR part 5

    Output columns
    --------------
    subject, visit, domain, aggregation, analysis_date, metric,
    python_value, ggir_value, difference, abs_difference
    """
    circadian_df = _build_circadian_rows(
        subject=subject,
        visit=visit,
        circadian_results=circadian_results,
        ggir_part6_path=ggir_part6_path,
    )
    activity_df = _build_activity_rows(
        subject=subject,
        visit=visit,
        intensity_daily_df=activity_results["intensity_daily_df"],
        ggir_part5_path=ggir_part5_path,
    )
    out = pd.concat([circadian_df, activity_df], ignore_index=True, sort=False)
    return out.sort_values(["domain", "aggregation", "analysis_date", "metric"]).reset_index(drop=True)


def pivot_pipeline_comparison_dataframe(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional wide-format view of the tidy comparison dataframe.
    """
    wide = (
        comparison_df.pivot_table(
            index=["subject", "visit", "domain", "aggregation", "analysis_date"],
            columns="metric",
            values=["python_value", "ggir_value", "difference", "abs_difference"],
            aggfunc="first",
        )
        .sort_index(axis=1)
    )
    wide.columns = [f"{value_type}__{metric}" for value_type, metric in wide.columns]
    return wide.reset_index()


def run_pipelines_and_build_comparison(subject: str,
                                       raw_root: str | Path,
                                       ggir_root: str | Path,
                                       sleep_csv_path: str | Path,
                                       visit: str = "T0",
                                       circadian_save_folder: Optional[str | Path] = None,
                                       nonwear_method: str = "vanhees2013",
                                       dayborder_hours: float = 0.0,
                                       min_wear_hours: float = 18.0,
                                       min_valid_days: int = 3,
                                       max_days: Optional[int] = 7,
                                       epoch_seconds: int = 5,
                                       sleep_subject: Optional[str] = None) -> Dict[str, object]:
    """
    Convenience wrapper that runs both Python pipelines and returns the combined
    comparison dataframe alongside the underlying results.
    """
    raw_file_path = find_wrist_accelerometer_file(raw_root=raw_root, subject=subject, visit=visit)
    ggir_files = find_ggir_metric_files(ggir_root=ggir_root, subject=subject, visit=visit)

    if circadian_save_folder is None:
        circadian_save_folder = raw_file_path.parent / "python_circadian_output"

    sleep_windows = load_sleep_windows(
        sleep_csv_path=sleep_csv_path,
        subject=subject,
        visit=visit,
        sleep_subject=sleep_subject,
    )

    circadian_results = circadian_pipeline(
        file_path=str(raw_file_path),
        fs=100,
        device="GENEActiv",
        save_folder=str(circadian_save_folder),
        nonwear_method=nonwear_method,
        dayborder_hours=dayborder_hours,
        min_wear_hours=min_wear_hours,
        max_days=max_days,
        min_valid_days=min_valid_days,
        plot=False,
    )
    activity_results = run_activity_intensity_pipeline(
        raw_root=raw_root,
        subject=subject,
        sleep_csv_path=sleep_csv_path,
        visit=visit,
        raw_file_path=raw_file_path,
        sleep_subject=sleep_subject,
        nonwear_method=nonwear_method,
        epoch_seconds=epoch_seconds,
        dayborder_hours=dayborder_hours,
        min_wear_hours=min_wear_hours,
        min_valid_days=min_valid_days,
        max_days=max_days,
    )

    comparison_df = build_pipeline_comparison_dataframe(
        subject=subject,
        visit=visit,
        circadian_results=circadian_results,
        activity_results=activity_results,
        ggir_part6_path=ggir_files["ms6_rdata_path"],
        ggir_part5_path=ggir_files["ms5_rdata_path"],
    )

    return {
        "raw_file_path": raw_file_path,
        "sleep_windows": sleep_windows,
        "ggir_files": ggir_files,
        "circadian_results": circadian_results,
        "activity_results": activity_results,
        "comparison_df": comparison_df,
        "comparison_wide_df": pivot_pipeline_comparison_dataframe(comparison_df),
    }


def write_comparison_outputs(comparison_df: pd.DataFrame,
                             output_dir: str | Path,
                             wide_df: Optional[pd.DataFrame] = None) -> Dict[str, Path]:
    """
    Write tidy and wide comparison tables to CSV.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tidy_path = output_dir / "pipeline_vs_ggir_comparison_long.csv"
    comparison_df.to_csv(tidy_path, index=False)

    if wide_df is None:
        wide_df = pivot_pipeline_comparison_dataframe(comparison_df)
    wide_path = output_dir / "pipeline_vs_ggir_comparison_wide.csv"
    wide_df.to_csv(wide_path, index=False)

    return {
        "comparison_long_csv": tidy_path,
        "comparison_wide_csv": wide_path,
    }
