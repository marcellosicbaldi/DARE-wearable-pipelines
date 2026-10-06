from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from dare_wearables.wrist.circadian.activity_intensity_empatica import (
    DEFAULT_GP_SLEEP_CSV,
    run_activity_intensity_pipeline_gp_from_silver,
    write_activity_intensity_outputs,
)
from dare_wearables.wrist.circadian.metrics import compare_against_ggir, load_ggir_part5_intensity
from dare_wearables.wrist.circadian.empatica import circadian_pipeline_gp_from_silver
from dare_wearables.wrist.circadian.pipeline_comparison import INTENSITY_METRICS


PART6_METRICS = [
    "cosinor_mes",
    "cosinor_amp",
    "cosinor_acrophase",
    "cosinor_acrotime",
    "cosinor_days",
    "IS",
    "IV",
    "L5VALUE",
    "M5VALUE",
    "L5TIME_num",
    "M5TIME_num",
    "L5TIME_clock",
    "M5TIME_clock",
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

    target = _normalize_subject_token(subject)
    matches = [
        path for path in root.iterdir()
        if path.is_dir() and _normalize_subject_token(path.name) == target
    ]
    if not matches:
        raise FileNotFoundError(f"Could not find subject {subject!r} under {root}.")
    if len(matches) > 1:
        raise ValueError(
            f"Found multiple matching subject folders for {subject!r}: {[path.name for path in matches]}"
        )
    return matches[0]


def _find_single_rdata_or_none(folder: str | Path) -> Optional[Path]:
    folder = Path(folder)
    files = sorted(folder.glob("*.RData"))
    if not files:
        return None
    if len(files) > 1:
        raise ValueError(f"Found multiple `.RData` files in {folder}: {[path.name for path in files]}")
    return files[0]


def find_gp_ggir_metric_files(silver_root: str | Path,
                              subject: str,
                              visit: str = "T0",
                              sensor: str = "Empatica",
                              ggir_folder: str = "output_fallspredict") -> Dict[str, Optional[Path]]:
    """
    Locate GP GGIR metric files under the configured silver layout:
    `{silver_root}/{subject}/{visit}/{sensor}/{ggir_folder}`

    Note: in the current GP Empatica runs, `ms5.out` is typically available but
    `ms6.out` is often absent because GGIR part 6 was disabled.
    """
    subject_dir = _resolve_subject_directory(silver_root, subject)
    base = subject_dir / str(visit) / str(sensor) / ggir_folder / "meta"
    if not base.exists():
        raise FileNotFoundError(f"GGIR output folder not found: {base}")

    ms5_path = _find_single_rdata_or_none(base / "ms5.out")
    if ms5_path is None:
        raise FileNotFoundError(f"No `ms5.out` RData file found in {base / 'ms5.out'}.")
    ms6_path = _find_single_rdata_or_none(base / "ms6.out")

    return {
        "ms5_rdata_path": ms5_path,
        "ms6_rdata_path": ms6_path,
    }


def _build_circadian_rows_gp(subject: str,
                             visit: str,
                             circadian_results: Dict[str, object],
                             ggir_part6_path: Optional[str | Path]) -> pd.DataFrame:
    if ggir_part6_path is not None:
        comparison_df = compare_against_ggir(
            metrics_df=circadian_results["metrics_df"],
            ggir_rdata_path=str(ggir_part6_path),
        ).copy()
        comparison_df["subject"] = str(subject)
        comparison_df["visit"] = str(visit)
        comparison_df["domain"] = "circadian"
        comparison_df["aggregation"] = "summary"
        comparison_df["analysis_date"] = pd.NaT
        comparison_df["ggir_available"] = True
        comparison_df = comparison_df.rename(columns={"python": "python_value", "ggir": "ggir_value"})
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
                "ggir_available",
            ]
        ]

    python_metrics = circadian_results["metrics_df"].iloc[0]
    rows = []
    for metric in PART6_METRICS:
        rows.append(
            {
                "subject": str(subject),
                "visit": str(visit),
                "domain": "circadian",
                "aggregation": "summary",
                "analysis_date": pd.NaT,
                "metric": metric,
                "python_value": python_metrics.get(metric, np.nan),
                "ggir_value": np.nan,
                "difference": np.nan,
                "abs_difference": np.nan,
                "ggir_available": False,
            }
        )
    return pd.DataFrame(rows)


def _build_activity_rows_gp(subject: str,
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
            py_metric = f"{metric}_python"
            ggir_metric = f"{metric}_ggir"
            rows.append(
                {
                    "subject": str(subject),
                    "visit": str(visit),
                    "domain": "activity_intensity",
                    "aggregation": "day",
                    "analysis_date": pd.to_datetime(row["analysis_date"]),
                    "metric": metric,
                    "python_value": float(row[py_metric]),
                    "ggir_value": float(row[ggir_metric]),
                    "difference": float(row[py_metric] - row[ggir_metric]),
                    "abs_difference": float(abs(row[py_metric] - row[ggir_metric])),
                    "ggir_available": True,
                }
            )

    if not rows:
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
                "ggir_available",
            ]
        )

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
    summary_df["ggir_available"] = True
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
            "ggir_available",
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
            "ggir_available",
        ]
    ]
    return pd.concat([summary_df, per_day_df], ignore_index=True)


def build_pipeline_comparison_dataframe_gp(subject: str,
                                           visit: str,
                                           circadian_results: Dict[str, object],
                                           activity_results: Dict[str, object],
                                           ggir_part5_path: str | Path,
                                           ggir_part6_path: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Create one tidy comparison dataframe for the GP dataset.

    Columns
    -------
    subject, visit, domain, aggregation, analysis_date, metric,
    python_value, ggir_value, difference, abs_difference, ggir_available
    """
    circadian_df = _build_circadian_rows_gp(
        subject=subject,
        visit=visit,
        circadian_results=circadian_results,
        ggir_part6_path=ggir_part6_path,
    )
    activity_df = _build_activity_rows_gp(
        subject=subject,
        visit=visit,
        intensity_daily_df=activity_results["intensity_daily_df"],
        ggir_part5_path=ggir_part5_path,
    )
    out = pd.concat([circadian_df, activity_df], ignore_index=True, sort=False)
    return out.sort_values(["domain", "aggregation", "analysis_date", "metric"]).reset_index(drop=True)


def pivot_pipeline_comparison_dataframe_gp(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional wide-format view of the GP comparison dataframe.
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


def run_pipelines_and_build_comparison_gp(subject: str,
                                          silver_root: str | Path,
                                          sleep_csv_path: str | Path = DEFAULT_GP_SLEEP_CSV,
                                          visit: str = "T0",
                                          sensor: str = "Empatica",
                                          circadian_save_folder: Optional[str | Path] = None,
                                          intensity_save_folder: Optional[str | Path] = None,
                                          nonwear_method: str = "empatica_detach",
                                          dayborder_hours: float = 0.0,
                                          min_wear_hours: float = 18.0,
                                          min_valid_days: int = 3,
                                          max_days: Optional[int] = 7,
                                          epoch_seconds: int = 5) -> Dict[str, object]:
    """
    Run the GP circadian and activity-intensity pipelines and return one combined
    comparison dataframe against the GP GGIR outputs.
    """
    ggir_files = find_gp_ggir_metric_files(
        silver_root=silver_root,
        subject=subject,
        visit=visit,
        sensor=sensor,
    )

    subject_dir = _resolve_subject_directory(silver_root, subject)
    sensor_dir = subject_dir / str(visit) / str(sensor)
    if circadian_save_folder is None:
        circadian_save_folder = sensor_dir / "python_circadian_output"
    if intensity_save_folder is None:
        intensity_save_folder = sensor_dir / "python_activity_intensity_output"

    circadian_results = circadian_pipeline_gp_from_silver(
        silver_root=silver_root,
        participant=subject,
        visit=visit,
        sensor=sensor,
        save_folder=str(circadian_save_folder),
        nonwear_method=nonwear_method,
        dayborder_hours=dayborder_hours,
        min_wear_hours=min_wear_hours,
        min_valid_days=min_valid_days,
        max_days=max_days,
        plot=False,
    )
    activity_results = run_activity_intensity_pipeline_gp_from_silver(
        silver_root=silver_root,
        participant=subject,
        visit=visit,
        sensor=sensor,
        sleep_csv_path=sleep_csv_path,
        nonwear_method=nonwear_method,
        dayborder_hours=dayborder_hours,
        min_wear_hours=min_wear_hours,
        min_valid_days=min_valid_days,
        max_days=max_days,
        epoch_seconds=epoch_seconds,
    )
    intensity_output_paths = write_activity_intensity_outputs(
        results=activity_results,
        output_dir=intensity_save_folder,
        include_epoch_table=False,
    )

    comparison_df = build_pipeline_comparison_dataframe_gp(
        subject=subject,
        visit=visit,
        circadian_results=circadian_results,
        activity_results=activity_results,
        ggir_part5_path=ggir_files["ms5_rdata_path"],
        ggir_part6_path=ggir_files["ms6_rdata_path"],
    )

    return {
        "ggir_files": ggir_files,
        "circadian_results": circadian_results,
        "activity_results": activity_results,
        "activity_output_paths": intensity_output_paths,
        "comparison_df": comparison_df,
        "comparison_wide_df": pivot_pipeline_comparison_dataframe_gp(comparison_df),
    }


def write_comparison_outputs_gp(comparison_df: pd.DataFrame,
                                output_dir: str | Path,
                                wide_df: Optional[pd.DataFrame] = None) -> Dict[str, Path]:
    """
    Write tidy and wide GP comparison tables to CSV.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tidy_path = output_dir / "pipeline_vs_ggir_gp_comparison_long.csv"
    comparison_df.to_csv(tidy_path, index=False)

    if wide_df is None:
        wide_df = pivot_pipeline_comparison_dataframe_gp(comparison_df)
    wide_path = output_dir / "pipeline_vs_ggir_gp_comparison_wide.csv"
    wide_df.to_csv(wide_path, index=False)

    return {
        "comparison_long_csv": tidy_path,
        "comparison_wide_csv": wide_path,
    }
