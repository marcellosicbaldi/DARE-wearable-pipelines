from __future__ import annotations

from dare_wearables.common.output_state import fresh_outputs

import os
import re
from pathlib import Path
from typing import Dict, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import polars as pl

from dare_wearables.wrist.circadian.activity_intensity_empatica import (
    run_activity_intensity_pipeline_gp_from_preprocessed,
    write_activity_intensity_outputs,
)
from dare_wearables.wrist.circadian.empatica import (
    find_empatica_parquet_files,
    preprocess_empatica_recording,
    run_circadian_pipeline_gp_from_preprocessed,
)
from dare_wearables.wrist.sleep.sleep_functions import compute_sleep_from_guider
from dare_wearables.wrist.sleep.vh2015_sib import vh2015_sib
from dare_wearables.wrist.sleep.vh2018_spt import vh2018_spt


_MIN_VALID_DAYS_RE = re.compile(
    r"Only (?P<n_valid_days>\d+) valid days found, but min_valid_days=(?P<min_valid_days>\d+)\."
)


def _infer_participant_from_acc_path(acc_parquet_path: str | Path) -> str:
    path = Path(acc_parquet_path)
    if len(path.parents) < 3:
        raise ValueError(f"Could not infer participant from path: {acc_parquet_path}")
    return path.parents[2].name


def _normalize_subject_token(value: object) -> str:
    text = str(value).strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not extract a numeric subject identifier from {value!r}.")
    return str(int(digits))


def _safe_normalize_subject_token(value: object) -> str | None:
    try:
        return _normalize_subject_token(value)
    except ValueError:
        return None


def _pick_column(columns: set[str],
                 explicit: str | None,
                 candidates: list[str],
                 *,
                 role: str,
                 csv_path: Path,
                 allow_missing: bool = False) -> str | None:
    if explicit is not None:
        if explicit not in columns:
            raise ValueError(
                f"Column {explicit!r} was requested as the {role} column, but it was not found in {csv_path}."
            )
        return explicit

    for candidate in candidates:
        if candidate in columns:
            return candidate

    if allow_missing:
        return None

    raise ValueError(
        f"Could not detect the {role} column in {csv_path}. "
        f"Tried: {candidates}"
    )


def _coerce_like_reference(series: pd.Series,
                           *,
                           reference_series: pd.Series | None,
                           timezone: str) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    if reference_series is None:
        ref_tz = None
    else:
        ref_ts = pd.to_datetime(reference_series, errors="coerce")
        ref_tz = getattr(ref_ts.dt, "tz", None)

    if ref_tz is None:
        if getattr(ts.dt, "tz", None) is None:
            return ts
        return ts.dt.tz_convert(ZoneInfo(timezone)).dt.tz_localize(None)

    if getattr(ts.dt, "tz", None) is None:
        return _localize_datetime_series(ts, ref_tz)
    return ts.dt.tz_convert(ref_tz)


def _empty_guider_windows() -> pd.DataFrame:
    return pd.DataFrame(columns=["subject", "visit", "night_id", "guider_start", "guider_end"])


def _localize_datetime_series(ts: pd.Series, timezone) -> pd.Series:
    try:
        return ts.dt.tz_localize(
            timezone,
            ambiguous="infer",
            nonexistent=pd.Timedelta(hours=1),
        )
    except Exception as exc:
        if exc.__class__.__name__ != "AmbiguousTimeError":
            raise
        return ts.dt.tz_localize(
            timezone,
            ambiguous=False,
            nonexistent=pd.Timedelta(hours=1),
        )


def _load_optional_guider_csv(csv_path: str | Path | None,
                              *,
                              participant: str,
                              visit: str,
                              timezone: str,
                              reference_series: pd.Series | None,
                              subject_col: str | None = None,
                              start_col: str | None = None,
                              end_col: str | None = None,
                              visit_col: str | None = None,
                              night_id_col: str | None = None,
                              allow_missing_subject: bool = False,
                              allow_missing_file: bool = False) -> pd.DataFrame:
    if csv_path is None:
        return _empty_guider_windows()

    csv_path = Path(csv_path)
    if not csv_path.exists():
        if allow_missing_file:
            return _empty_guider_windows()
        raise FileNotFoundError(f"Guider CSV not found: {csv_path}")

    guider_df = pd.read_csv(csv_path)
    columns = set(guider_df.columns)

    subject_col = _pick_column(
        columns,
        subject_col,
        ["id", "subject", "participant", "participant_id"],
        role="subject",
        csv_path=csv_path,
        allow_missing=allow_missing_subject,
    )
    start_col = _pick_column(
        columns,
        start_col,
        ["Datetime_bed", "tib_start", "start", "guider_start", "start_time", "start_datetime", "diary_start"],
        role="start",
        csv_path=csv_path,
    )
    end_col = _pick_column(
        columns,
        end_col,
        ["Datetime_wake", "tib_end", "end", "guider_end", "end_time", "end_datetime", "diary_end"],
        role="end",
        csv_path=csv_path,
    )
    visit_col = _pick_column(
        columns,
        visit_col,
        ["visit", "timepoint"],
        role="visit",
        csv_path=csv_path,
        allow_missing=True,
    )
    night_id_col = _pick_column(
        columns,
        night_id_col,
        ["night_id", "night"],
        role="night ID",
        csv_path=csv_path,
        allow_missing=True,
    )

    target_subject = str(participant).strip()

    if subject_col is None:
        match_df = guider_df.copy()
        if visit_col is not None:
            match_df[visit_col] = match_df[visit_col].astype(str).str.strip()
            match_df = match_df.loc[match_df[visit_col] == visit].copy()
    else:
        guider_df[subject_col] = guider_df[subject_col].astype(str).str.strip()
        match_df = guider_df.loc[guider_df[subject_col] == target_subject].copy()
        if visit_col is not None:
            match_df[visit_col] = match_df[visit_col].astype(str).str.strip()
            match_df = match_df.loc[match_df[visit_col] == visit].copy()

    if match_df.empty and subject_col is not None:
        target_norm = _safe_normalize_subject_token(target_subject)
        if target_norm is not None:
            candidates = guider_df.copy()
            if visit_col is not None:
                candidates[visit_col] = candidates[visit_col].astype(str).str.strip()
                candidates = candidates.loc[candidates[visit_col] == visit].copy()
            candidates["_subject_norm"] = candidates[subject_col].map(_safe_normalize_subject_token)
            match_df = candidates.loc[candidates["_subject_norm"] == target_norm].copy()

    if match_df.empty:
        return _empty_guider_windows()

    match_df["guider_start"] = _coerce_like_reference(
        match_df[start_col],
        reference_series=reference_series,
        timezone=timezone,
    )
    match_df["guider_end"] = _coerce_like_reference(
        match_df[end_col],
        reference_series=reference_series,
        timezone=timezone,
    )
    match_df = match_df.loc[
        match_df["guider_start"].notna() &
        match_df["guider_end"].notna() &
        (match_df["guider_end"] > match_df["guider_start"])
    ].copy()
    match_df = match_df.sort_values("guider_start").reset_index(drop=True)

    if match_df.empty:
        return _empty_guider_windows()

    if night_id_col is not None:
        night_id = match_df[night_id_col].to_numpy()
    else:
        night_id = np.arange(1, len(match_df) + 1)

    visit_values = match_df[visit_col].astype(str) if visit_col is not None else pd.Series([visit] * len(match_df))
    subject_values = (
        match_df[subject_col].astype(str).to_numpy()
        if subject_col is not None
        else np.repeat(target_subject, len(match_df))
    )

    return pd.DataFrame({
        "subject": subject_values,
        "visit": visit_values.to_numpy(),
        "night_id": night_id,
        "guider_start": match_df["guider_start"].to_numpy(),
        "guider_end": match_df["guider_end"].to_numpy(),
    })


def _empty_sleep_output() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "night_id", "guider_start", "guider_end", "guider_dur",
        "spt_start", "spt_end", "spt_dur", "nonwear_dur_in_guider",
        "n_bouts", "TST", "longest_bout", "sleep_onset_proxy", "wake_time_proxy",
        "n_awakenings", "WASO", "max_wake_gap", "sleep_midpoint", "sleep_efficiency",
        "awakenings_per_hour_spt", "fragmentation_awakenings_per_hour_tst", "spt_found",
    ])


def _empty_sib_in_guider() -> pd.DataFrame:
    return pd.DataFrame(columns=["night_id", "start", "end", "duration"])


def _run_sleep_for_guider_source(guider_source: str,
                                 guider_windows_df: pd.DataFrame,
                                 *,
                                 guider_start_col: str,
                                 guider_end_col: str,
                                 guider_id_col: str | None,
                                 sib_bouts_df: pd.DataFrame,
                                 nonwear_df: pd.DataFrame,
                                 sib_assignment_mode: str = "overlap") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    guider_windows_out = guider_windows_df.copy()
    if guider_windows_out.empty:
        sleep_output = _empty_sleep_output()
        sib_in_guider = _empty_sib_in_guider()
    else:
        sleep_output, sib_in_guider = compute_sleep_from_guider(
            guider_df=guider_windows_out,
            sib_bouts=sib_bouts_df,
            nonwear_periods_final=nonwear_df,
            guider_start_col=guider_start_col,
            guider_end_col=guider_end_col,
            guider_id_col=guider_id_col,
            sib_start_col="start",
            sib_end_col="end",
            nw_start_col="start",
            nw_end_col="end",
            sib_assignment_mode=sib_assignment_mode,
        )

    guider_windows_out["guider_source"] = guider_source
    sleep_output["guider_source"] = guider_source
    sib_in_guider["guider_source"] = guider_source
    return guider_windows_out, sleep_output, sib_in_guider


def _source_filename_token(guider_source: str) -> str:
    return guider_source.lower().replace(" ", "_")


def _intervals_overlap(start_a: pd.Timestamp,
                       end_a: pd.Timestamp,
                       start_b: pd.Timestamp,
                       end_b: pd.Timestamp) -> bool:
    return bool((start_a < end_b) and (end_a > start_b))


def _select_activity_sleep_windows(sleep_outputs_by_source: Dict[str, pd.DataFrame],
                                   *,
                                   participant: str,
                                   visit: str,
                                   timezone: str = "Europe/Rome") -> pd.DataFrame:
    """
    Build the SPT table used for activity-intensity summaries.

    Priority order
    --------------
    1. diary-derived SPT from the overlap between SIB and diary guider
    2. HDCZA-derived SPT for nights not already covered by a diary-derived SPT

    Lower-back TIB is still summarized in the sleep outputs, but it is not used
    here because the activity-intensity pipeline needs an SPT rather than a
    broader in-bed interval.
    """
    diary_df = sleep_outputs_by_source.get("sleep_diary", pd.DataFrame()).copy()
    hdcza_df = sleep_outputs_by_source.get("HDCZA", pd.DataFrame()).copy()

    selected_rows: list[dict[str, object]] = []
    diary_cover_windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    if not diary_df.empty:
        diary_df = diary_df.loc[diary_df["spt_found"].fillna(False)].copy()
        diary_df = diary_df.sort_values("guider_start").reset_index(drop=True)
        for row in diary_df.itertuples(index=False):
            selected_rows.append({
                "source_night_id": row.night_id,
                "spt_start": row.spt_start,
                "spt_end": row.spt_end,
                "sleep_window_source": "sleep_diary",
                "priority_rank": 1,
            })
            diary_cover_windows.append((row.guider_start, row.guider_end))

    if not hdcza_df.empty:
        hdcza_df = hdcza_df.loc[hdcza_df["spt_found"].fillna(False)].copy()
        hdcza_df = hdcza_df.sort_values("guider_start").reset_index(drop=True)
        for row in hdcza_df.itertuples(index=False):
            overlaps_diary = any(
                _intervals_overlap(row.guider_start, row.guider_end, start, end)
                for start, end in diary_cover_windows
            )
            if overlaps_diary:
                continue
            selected_rows.append({
                "source_night_id": row.night_id,
                "spt_start": row.spt_start,
                "spt_end": row.spt_end,
                "sleep_window_source": "HDCZA",
                "priority_rank": 2,
            })

    if not selected_rows:
        return pd.DataFrame(
            columns=[
                "subject",
                "visit",
                "night_id",
                "source_night_id",
                "sleep_window_source",
                "priority_rank",
                "spt_start",
                "spt_end",
            ]
        )

    out = pd.DataFrame(selected_rows)
    out = out.sort_values("spt_start").reset_index(drop=True)
    out["night_id"] = np.arange(1, len(out) + 1)
    out["subject"] = str(participant)
    out["visit"] = str(visit)
    for col in ("spt_start", "spt_end"):
        ts = pd.to_datetime(out[col], errors="coerce")
        if getattr(ts.dt, "tz", None) is None:
            out[col] = _localize_datetime_series(ts, ZoneInfo(timezone))
        else:
            out[col] = ts.dt.tz_convert(ZoneInfo(timezone))
    return out[
        [
            "subject",
            "visit",
            "night_id",
            "source_night_id",
            "sleep_window_source",
            "priority_rank",
            "spt_start",
            "spt_end",
        ]
    ]


def _is_min_valid_days_error(exc: ValueError) -> bool:
    return _MIN_VALID_DAYS_RE.search(str(exc)) is not None


def _skipped_stage_result(stage: str, exc: ValueError) -> dict[str, object]:
    match = _MIN_VALID_DAYS_RE.search(str(exc))
    out: dict[str, object] = {
        "stage": stage,
        "status": "skipped",
        "reason": str(exc),
    }
    if match is not None:
        out["n_valid_days"] = int(match.group("n_valid_days"))
        out["min_valid_days"] = int(match.group("min_valid_days"))
    return out


def _skipped_stage_reason(stage: str, reason: str) -> dict[str, object]:
    return {
        "stage": stage,
        "status": "skipped",
        "reason": reason,
    }


@fresh_outputs(["sleep_*.csv", "recording_info.csv"])
def run_sleep_pipeline_gp_from_preprocessed(calibrated_df: pl.DataFrame,
                                            acc_df: pd.DataFrame,
                                            temp_df: pd.DataFrame,
                                            info: dict,
                                            nonwear_df: pd.DataFrame,
                                            *,
                                            participant: str,
                                            visit: str = "T0",
                                            timezone: str = "Europe/Rome",
                                            diary_csv_path: Optional[str | Path] = None,
                                            diary_subject_col: str | None = None,
                                            diary_start_col: str | None = None,
                                            diary_end_col: str | None = None,
                                            diary_visit_col: str | None = None,
                                            diary_night_id_col: str | None = None,
                                            lb_tib_csv_path: Optional[str | Path] = None,
                                            lb_subject_col: str | None = None,
                                            lb_start_col: str | None = None,
                                            lb_end_col: str | None = None,
                                            lb_visit_col: str | None = None,
                                            lb_night_id_col: str | None = None,
                                            sib_change_thresh_deg: float = 5.0,
                                            sib_min_bout_minutes: int = 5,
                                            haspt_ignore_invalid: float | bool = False,
                                            hdcza_threshold: float | tuple[float, float] | list[float] | None = None,
                                            spt_min_block_dur_minutes: int = 30,
                                            spt_max_gap_dur_minutes: int = 60,
                                            spt_max_gap_ratio: float = 1.0,
                                            rerun_with_6pm_window: bool = True,
                                            return_epoch_outputs: bool = True,
                                            return_intermediates: bool = False,
                                            save_folder: Optional[str | Path] = None) -> Dict[str, object]:
    """
    Run the GP sleep pipeline from a shared preprocessed recording.
    """
    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)

    nonwear_pl = pl.from_pandas(nonwear_df) if isinstance(nonwear_df, pd.DataFrame) and not nonwear_df.empty else None

    sib_out = vh2015_sib(
        calibrated_df,
        nonwear_df=nonwear_pl,
        ts_col="time",
        x_col="x_cal",
        y_col="y_cal",
        z_col="z_cal",
        change_thresh_deg=sib_change_thresh_deg,
        min_bout_minutes=sib_min_bout_minutes,
        return_epochs=return_epoch_outputs,
        return_intermediates=return_intermediates,
    )
    spt_out = vh2018_spt(
        calibrated_df,
        nonwear_df=nonwear_pl,
        ts_col="time",
        x_col="x_cal",
        y_col="y_cal",
        z_col="z_cal",
        epoch_seconds=5,
        haspt_ignore_invalid=haspt_ignore_invalid,
        hdcza_threshold=hdcza_threshold,
        spt_min_block_dur_minutes=spt_min_block_dur_minutes,
        spt_max_gap_dur_minutes=spt_max_gap_dur_minutes,
        spt_max_gap_ratio=spt_max_gap_ratio,
        rerun_with_6pm_window=rerun_with_6pm_window,
        return_epochs=return_epoch_outputs,
        return_intermediates=return_intermediates,
    )

    spt_windows_df = spt_out["spt_windows"].to_pandas()
    sib_bouts_df = sib_out["sib_bouts"].to_pandas()
    time_reference = sib_bouts_df["start"] if not sib_bouts_df.empty else pd.Series(calibrated_df["time"].to_list())

    diary_guider_df = _load_optional_guider_csv(
        diary_csv_path,
        participant=participant,
        visit=visit,
        timezone=timezone,
        reference_series=time_reference,
        subject_col=diary_subject_col,
        start_col=diary_start_col,
        end_col=diary_end_col,
        visit_col=diary_visit_col,
        night_id_col=diary_night_id_col,
    )
    lb_guider_df = _load_optional_guider_csv(
        lb_tib_csv_path,
        participant=participant,
        visit=visit,
        timezone=timezone,
        reference_series=time_reference,
        subject_col=lb_subject_col,
        start_col=lb_start_col,
        end_col=lb_end_col,
        visit_col=lb_visit_col,
        night_id_col=lb_night_id_col,
        allow_missing_subject=True,
        allow_missing_file=True,
    )

    guider_windows_by_source: Dict[str, pd.DataFrame] = {}
    sleep_outputs_by_source: Dict[str, pd.DataFrame] = {}
    sib_in_guider_by_source: Dict[str, pd.DataFrame] = {}

    hdcza_guider_windows_df, sleep_output_df, sib_in_guider_df = _run_sleep_for_guider_source(
        "HDCZA",
        spt_windows_df,
        guider_start_col="spt_start",
        guider_end_col="spt_end",
        guider_id_col="night_id",
        sib_bouts_df=sib_bouts_df,
        nonwear_df=nonwear_df,
    )
    guider_windows_by_source["HDCZA"] = hdcza_guider_windows_df
    sleep_outputs_by_source["HDCZA"] = sleep_output_df
    sib_in_guider_by_source["HDCZA"] = sib_in_guider_df

    diary_windows_df, diary_sleep_output_df, diary_sib_in_guider_df = _run_sleep_for_guider_source(
        "sleep_diary",
        diary_guider_df,
        guider_start_col="guider_start",
        guider_end_col="guider_end",
        guider_id_col="night_id",
        sib_bouts_df=sib_bouts_df,
        nonwear_df=nonwear_df,
    )
    guider_windows_by_source["sleep_diary"] = diary_windows_df
    sleep_outputs_by_source["sleep_diary"] = diary_sleep_output_df
    sib_in_guider_by_source["sleep_diary"] = diary_sib_in_guider_df

    lb_windows_df, lb_sleep_output_df, lb_sib_in_guider_df = _run_sleep_for_guider_source(
        "lower_back_tib",
        lb_guider_df,
        guider_start_col="guider_start",
        guider_end_col="guider_end",
        guider_id_col="night_id",
        sib_bouts_df=sib_bouts_df,
        nonwear_df=nonwear_df,
        sib_assignment_mode="contained",
    )
    guider_windows_by_source["lower_back_tib"] = lb_windows_df
    sleep_outputs_by_source["lower_back_tib"] = lb_sleep_output_df
    sib_in_guider_by_source["lower_back_tib"] = lb_sib_in_guider_df

    guider_windows_all_df = pd.concat(
        [df for df in guider_windows_by_source.values() if not df.empty],
        ignore_index=True,
        sort=False,
    ) if any(not df.empty for df in guider_windows_by_source.values()) else pd.DataFrame()
    sleep_output_all_df = pd.concat(
        [df for df in sleep_outputs_by_source.values() if not df.empty],
        ignore_index=True,
        sort=False,
    ) if any(not df.empty for df in sleep_outputs_by_source.values()) else pd.DataFrame()
    sib_in_guider_all_df = pd.concat(
        [df for df in sib_in_guider_by_source.values() if not df.empty],
        ignore_index=True,
        sort=False,
    ) if any(not df.empty for df in sib_in_guider_by_source.values()) else pd.DataFrame()

    info = dict(info)
    info["n_hdcza_guider_windows"] = int(len(spt_windows_df))
    info["n_diary_guider_windows"] = int(len(diary_guider_df))
    info["n_lower_back_guider_windows"] = int(len(lb_guider_df))
    info["hdcza_status"] = str(spt_out.get("status", "completed"))
    info["hdcza_reason"] = str(spt_out.get("reason", ""))
    info["available_guider_sources"] = [
        source for source, df in guider_windows_by_source.items()
        if not df.empty
    ]

    out: Dict[str, object] = {
        "participant": participant,
        "visit": visit,
        "info": info,
        "acc_df": acc_df,
        "temp_df": temp_df,
        "calibrated_df": calibrated_df,
        "nonwear_df": nonwear_df,
        "sib_bouts_df": sib_bouts_df,
        "spt_windows_df": spt_windows_df,
        "sleep_output_df": sleep_output_df,
        "sib_in_guider_df": sib_in_guider_df,
        "diary_guider_df": diary_guider_df,
        "lb_tib_guider_df": lb_guider_df,
        "guider_windows_by_source": guider_windows_by_source,
        "sleep_outputs_by_source": sleep_outputs_by_source,
        "sib_in_guider_by_source": sib_in_guider_by_source,
        "guider_windows_all_df": guider_windows_all_df,
        "sleep_output_all_df": sleep_output_all_df,
        "sib_in_guider_all_df": sib_in_guider_all_df,
        "sib_out": sib_out,
        "spt_out": spt_out,
    }

    if save_folder is not None:
        info_df = pd.DataFrame([info])
        info_df.to_csv(Path(save_folder) / "recording_info.csv", index=False)
        nonwear_df.to_csv(Path(save_folder) / "sleep_nonwear_windows.csv", index=False)
        sib_bouts_df.to_csv(Path(save_folder) / "sleep_sib_bouts.csv", index=False)
        spt_windows_df.to_csv(Path(save_folder) / "sleep_spt_windows.csv", index=False)
        sleep_output_df.to_csv(Path(save_folder) / "sleep_output.csv", index=False)
        sib_in_guider_df.to_csv(Path(save_folder) / "sleep_sib_in_guider.csv", index=False)
        if not guider_windows_all_df.empty:
            guider_windows_all_df.to_csv(Path(save_folder) / "sleep_guider_windows_all.csv", index=False)
        if not sleep_output_all_df.empty:
            sleep_output_all_df.to_csv(Path(save_folder) / "sleep_output_all_guiders.csv", index=False)
        if not sib_in_guider_all_df.empty:
            sib_in_guider_all_df.to_csv(Path(save_folder) / "sleep_sib_in_guider_all.csv", index=False)

        for guider_source, guider_windows_src_df in guider_windows_by_source.items():
            token = _source_filename_token(guider_source)
            sleep_outputs_by_source[guider_source].to_csv(
                Path(save_folder) / f"sleep_output_{token}.csv",
                index=False,
            )
            sib_in_guider_by_source[guider_source].to_csv(
                Path(save_folder) / f"sleep_sib_in_guider_{token}.csv",
                index=False,
            )
            guider_windows_src_df.to_csv(
                Path(save_folder) / f"sleep_guider_windows_{token}.csv",
                index=False,
            )

        if return_epoch_outputs:
            sib_epochs = sib_out.get("sib_epochs")
            if isinstance(sib_epochs, pl.DataFrame) and not sib_epochs.is_empty():
                sib_epochs.to_pandas().to_csv(Path(save_folder) / "sleep_sib_epochs.csv", index=False)
            spt_night_epochs = spt_out.get("spt_night_epochs")
            if isinstance(spt_night_epochs, pl.DataFrame) and not spt_night_epochs.is_empty():
                spt_night_epochs.to_pandas().to_csv(Path(save_folder) / "sleep_spt_night_epochs.csv", index=False)

    return out


@fresh_outputs(["sleep_*.csv", "recording_info.csv"])
def run_sleep_pipeline_gp(acc_parquet_path: str | Path,
                          temp_parquet_path: str | Path,
                          participant: Optional[str] = None,
                          visit: str = "T0",
                          recruitment_tracker_path: Optional[str | Path] = None,
                          parquet_engine: str = "fastparquet",
                          nonwear_method: str = "empatica_detach",
                          timezone: str = "Europe/Rome",
                          diary_csv_path: Optional[str | Path] = None,
                          diary_subject_col: str | None = None,
                          diary_start_col: str | None = None,
                          diary_end_col: str | None = None,
                          diary_visit_col: str | None = None,
                          diary_night_id_col: str | None = None,
                          lb_tib_csv_path: Optional[str | Path] = None,
                          lb_subject_col: str | None = None,
                          lb_start_col: str | None = None,
                          lb_end_col: str | None = None,
                          lb_visit_col: str | None = None,
                          lb_night_id_col: str | None = None,
                          sib_change_thresh_deg: float = 5.0,
                          sib_min_bout_minutes: int = 5,
                          haspt_ignore_invalid: float | bool = False,
                          hdcza_threshold: float | tuple[float, float] | list[float] | None = None,
                          spt_min_block_dur_minutes: int = 30,
                          spt_max_gap_dur_minutes: int = 60,
                          spt_max_gap_ratio: float = 1.0,
                          rerun_with_6pm_window: bool = True,
                          return_epoch_outputs: bool = True,
                          return_intermediates: bool = False,
                          save_folder: Optional[str | Path] = None) -> Dict[str, object]:
    """
    GP / Empatica sleep pipeline.

    This is the sleep-oriented analogue of `circadian_pipeline_gp.py`: it
    reuses the same preprocessing and then runs the sleep-specific algorithms.

    Steps
    -----
    1. load `acc.parquet` and `temp.parquet`
    2. autocalibrate the accelerometer stream
    3. detect non-wear with the same GP workflow used in the circadian modules:
       charging gaps + DETACH on non-charging chunks
    4. compute SIB on the calibrated acceleration with non-wear masked out
    5. compute the HDCZA SPT guider on the same calibrated acceleration
    6. optionally load diary and lower-back guider windows from CSV
    7. derive nightly sleep output from each available guider with `compute_sleep_from_guider`

    Non-wear handling
    -----------------
    The non-wear preprocessing is exactly the same as in the GP circadian
    pipeline, via `preprocess_empatica_recording()`:
    - charging gaps are treated as non-wear
    - DETACH is run on non-charging chunks
    - both sources are merged into final non-wear windows

    Those non-wear windows are then used for every guider scenario:
    - in `vh2015_sib`, non-wear angle epochs are masked, so SIB bouts break
      across invalid time
    - in `vh2018_spt`, non-wear defines invalid epochs for the HDCZA guider
      before the guider-specific invalid handling / imputation is applied
    - in diary / lower-back guided sleep summaries, the same non-wear windows
      are intersected with each guider window and reported as
      `nonwear_dur_in_guider`
    """
    if participant is None:
        participant = _infer_participant_from_acc_path(acc_parquet_path)

    preprocessed = preprocess_empatica_recording(
        acc_parquet_path=acc_parquet_path,
        temp_parquet_path=temp_parquet_path,
        parquet_engine=parquet_engine,
        recruitment_tracker_path=recruitment_tracker_path,
        nonwear_method=nonwear_method,
    )
    return run_sleep_pipeline_gp_from_preprocessed(
        calibrated_df=preprocessed["calibrated_df"],
        acc_df=preprocessed["acc_df"],
        temp_df=preprocessed["temp_df"],
        info=preprocessed["info"],
        nonwear_df=preprocessed["nonwear_df"],
        participant=participant,
        visit=visit,
        timezone=timezone,
        diary_csv_path=diary_csv_path,
        diary_subject_col=diary_subject_col,
        diary_start_col=diary_start_col,
        diary_end_col=diary_end_col,
        diary_visit_col=diary_visit_col,
        diary_night_id_col=diary_night_id_col,
        lb_tib_csv_path=lb_tib_csv_path,
        lb_subject_col=lb_subject_col,
        lb_start_col=lb_start_col,
        lb_end_col=lb_end_col,
        lb_visit_col=lb_visit_col,
        lb_night_id_col=lb_night_id_col,
        sib_change_thresh_deg=sib_change_thresh_deg,
        sib_min_bout_minutes=sib_min_bout_minutes,
        haspt_ignore_invalid=haspt_ignore_invalid,
        hdcza_threshold=hdcza_threshold,
        spt_min_block_dur_minutes=spt_min_block_dur_minutes,
        spt_max_gap_dur_minutes=spt_max_gap_dur_minutes,
        spt_max_gap_ratio=spt_max_gap_ratio,
        rerun_with_6pm_window=rerun_with_6pm_window,
        return_epoch_outputs=return_epoch_outputs,
        return_intermediates=return_intermediates,
        save_folder=save_folder,
    )


@fresh_outputs(["sleep_*.csv", "circadian_*.csv", "activity_intensity_*.csv",
                "recording_info.csv", "pipeline_stage_status.csv"])
def run_sleep_and_circadian_pipeline_gp(acc_parquet_path: str | Path,
                                        temp_parquet_path: str | Path,
                                        recruitment_tracker_path: Optional[str | Path] = None,
                                        participant: Optional[str] = None,
                                        visit: str = "T0",
                                        parquet_engine: str = "fastparquet",
                                        nonwear_method: str = "empatica_detach",
                                        timezone: str = "Europe/Rome",
                                        diary_csv_path: Optional[str | Path] = None,
                                        diary_subject_col: str | None = None,
                                        diary_start_col: str | None = None,
                                        diary_end_col: str | None = None,
                                        diary_visit_col: str | None = None,
                                        diary_night_id_col: str | None = None,
                                        lb_tib_csv_path: Optional[str | Path] = None,
                                        lb_subject_col: str | None = None,
                                        lb_start_col: str | None = None,
                                        lb_end_col: str | None = None,
                                        lb_visit_col: str | None = None,
                                        lb_night_id_col: str | None = None,
                                        sib_change_thresh_deg: float = 5.0,
                                        sib_min_bout_minutes: int = 5,
                                        haspt_ignore_invalid: float | bool = False,
                                        hdcza_threshold: float | tuple[float, float] | list[float] | None = None,
                                        spt_min_block_dur_minutes: int = 30,
                                        spt_max_gap_dur_minutes: int = 60,
                                        spt_max_gap_ratio: float = 1.0,
                                        rerun_with_6pm_window: bool = True,
                                        return_epoch_outputs: bool = True,
                                        return_intermediates: bool = False,
                                        dayborder_hours: float = 0.0,
                                        min_wear_hours: float = 18.0,
                                        max_nonwear_hours: Optional[float] = None,
                                        require_full_24h: bool = False,
                                        max_days: Optional[int] = 7,
                                        min_valid_days: Optional[int] = 3,
                                        activity_threshold_mg: float = 40.0,
                                        intensity_thresholds_mg: tuple[float, float, float] = (40.0, 100.0, 400.0),
                                        l5_m5_step_minutes: int = 10,
                                        activity_epoch_seconds: int = 5,
                                        require_activity_spt_overlap: bool = True,
                                        plot: bool = False,
                                        save_folder: Optional[str | Path] = None) -> Dict[str, object]:
    """
    Combined GP pipeline that runs sleep, circadian, and activity-intensity
    analyses after a single shared Empatica preprocessing step.

    Activity-intensity sleep windows are selected with this priority:
    1. diary-derived SPT from SIB ∩ diary guider
    2. HDCZA-derived SPT for nights not already covered by diary
    """
    if participant is None:
        participant = _infer_participant_from_acc_path(acc_parquet_path)

    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)

    preprocessed = preprocess_empatica_recording(
        acc_parquet_path=acc_parquet_path,
        temp_parquet_path=temp_parquet_path,
        parquet_engine=parquet_engine,
        nonwear_method=nonwear_method,
        recruitment_tracker_path=recruitment_tracker_path,
    )

    return run_wrist_from_preprocessed(
        preprocessed=preprocessed,
        participant=participant,
        visit=visit,
        nonwear_method=nonwear_method,
        timezone=timezone,
        diary_csv_path=diary_csv_path,
        diary_subject_col=diary_subject_col,
        diary_start_col=diary_start_col,
        diary_end_col=diary_end_col,
        diary_visit_col=diary_visit_col,
        diary_night_id_col=diary_night_id_col,
        lb_tib_csv_path=lb_tib_csv_path,
        lb_subject_col=lb_subject_col,
        lb_start_col=lb_start_col,
        lb_end_col=lb_end_col,
        lb_visit_col=lb_visit_col,
        lb_night_id_col=lb_night_id_col,
        sib_change_thresh_deg=sib_change_thresh_deg,
        sib_min_bout_minutes=sib_min_bout_minutes,
        haspt_ignore_invalid=haspt_ignore_invalid,
        hdcza_threshold=hdcza_threshold,
        spt_min_block_dur_minutes=spt_min_block_dur_minutes,
        spt_max_gap_dur_minutes=spt_max_gap_dur_minutes,
        spt_max_gap_ratio=spt_max_gap_ratio,
        rerun_with_6pm_window=rerun_with_6pm_window,
        return_epoch_outputs=return_epoch_outputs,
        return_intermediates=return_intermediates,
        dayborder_hours=dayborder_hours,
        min_wear_hours=min_wear_hours,
        max_nonwear_hours=max_nonwear_hours,
        require_full_24h=require_full_24h,
        max_days=max_days,
        min_valid_days=min_valid_days,
        activity_threshold_mg=activity_threshold_mg,
        intensity_thresholds_mg=intensity_thresholds_mg,
        l5_m5_step_minutes=l5_m5_step_minutes,
        activity_epoch_seconds=activity_epoch_seconds,
        require_activity_spt_overlap=require_activity_spt_overlap,
        plot=plot,
        save_folder=save_folder,
    )


@fresh_outputs(["sleep_*.csv", "circadian_*.csv", "activity_intensity_*.csv",
                "recording_info.csv", "pipeline_stage_status.csv"])
def run_wrist_from_preprocessed(preprocessed: dict, *,
    participant: str,
    visit: str = "T0",
    nonwear_method: str = "empatica_detach",
    timezone: str = "Europe/Rome",
    diary_csv_path: Optional[str | Path] = None,
    diary_subject_col: str | None = None,
    diary_start_col: str | None = None,
    diary_end_col: str | None = None,
    diary_visit_col: str | None = None,
    diary_night_id_col: str | None = None,
    lb_tib_csv_path: Optional[str | Path] = None,
    lb_subject_col: str | None = None,
    lb_start_col: str | None = None,
    lb_end_col: str | None = None,
    lb_visit_col: str | None = None,
    lb_night_id_col: str | None = None,
    sib_change_thresh_deg: float = 5.0,
    sib_min_bout_minutes: int = 5,
    haspt_ignore_invalid: float | bool = False,
    hdcza_threshold: float | tuple[float, float] | list[float] | None = None,
    spt_min_block_dur_minutes: int = 30,
    spt_max_gap_dur_minutes: int = 60,
    spt_max_gap_ratio: float = 1.0,
    rerun_with_6pm_window: bool = True,
    return_epoch_outputs: bool = True,
    return_intermediates: bool = False,
    dayborder_hours: float = 0.0,
    min_wear_hours: float = 18.0,
    max_nonwear_hours: Optional[float] = None,
    require_full_24h: bool = False,
    max_days: Optional[int] = 7,
    min_valid_days: Optional[int] = 3,
    activity_threshold_mg: float = 40.0,
    intensity_thresholds_mg: tuple[float, float, float] = (40.0, 100.0, 400.0),
    l5_m5_step_minutes: int = 10,
    activity_epoch_seconds: int = 5,
    require_activity_spt_overlap: bool = True,
    plot: bool = False,
    save_folder: Optional[str | Path] = None) -> Dict[str, object]:
    """Run sleep, circadian and activity stages from either device's prepared data.

    The sensor adapter supplies calibrated acceleration, raw acceleration,
    temperature, nonwear intervals and recording info in `preprocessed`.
    Processing thresholds, stage skip rules and output schemas are shared.
    """
    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)

    sleep_results = run_sleep_pipeline_gp_from_preprocessed(
        calibrated_df=preprocessed["calibrated_df"],
        acc_df=preprocessed["acc_df"],
        temp_df=preprocessed["temp_df"],
        info=preprocessed["info"],
        nonwear_df=preprocessed["nonwear_df"],
        participant=participant,
        visit=visit,
        timezone=timezone,
        diary_csv_path=diary_csv_path,
        diary_subject_col=diary_subject_col,
        diary_start_col=diary_start_col,
        diary_end_col=diary_end_col,
        diary_visit_col=diary_visit_col,
        diary_night_id_col=diary_night_id_col,
        lb_tib_csv_path=lb_tib_csv_path,
        lb_subject_col=lb_subject_col,
        lb_start_col=lb_start_col,
        lb_end_col=lb_end_col,
        lb_visit_col=lb_visit_col,
        lb_night_id_col=lb_night_id_col,
        sib_change_thresh_deg=sib_change_thresh_deg,
        sib_min_bout_minutes=sib_min_bout_minutes,
        haspt_ignore_invalid=haspt_ignore_invalid,
        hdcza_threshold=hdcza_threshold,
        spt_min_block_dur_minutes=spt_min_block_dur_minutes,
        spt_max_gap_dur_minutes=spt_max_gap_dur_minutes,
        spt_max_gap_ratio=spt_max_gap_ratio,
        rerun_with_6pm_window=rerun_with_6pm_window,
        return_epoch_outputs=return_epoch_outputs,
        return_intermediates=return_intermediates,
        save_folder=save_folder,
    )

    stage_status: list[dict[str, object]] = [
        {"stage": "sleep", "status": "completed", "reason": ""}
    ]

    try:
        circadian_results = run_circadian_pipeline_gp_from_preprocessed(
            calibrated_df=preprocessed["calibrated_df"],
            acc_df=preprocessed["acc_df"],
            temp_df=preprocessed["temp_df"],
            info=preprocessed["info"],
            nonwear_df=preprocessed["nonwear_df"],
            save_folder=save_folder,
            nonwear_method=nonwear_method,
            dayborder_hours=dayborder_hours,
            min_wear_hours=min_wear_hours,
            max_nonwear_hours=max_nonwear_hours,
            require_full_24h=require_full_24h,
            max_days=max_days,
            min_valid_days=min_valid_days,
            activity_threshold_mg=activity_threshold_mg,
            intensity_thresholds_mg=intensity_thresholds_mg,
            l5_m5_step_minutes=l5_m5_step_minutes,
            plot=plot,
        )
        stage_status.append({"stage": "circadian", "status": "completed", "reason": ""})
    except ValueError as exc:
        if not _is_min_valid_days_error(exc):
            raise
        circadian_results = _skipped_stage_result("circadian", exc)
        stage_status.append(circadian_results.copy())

    activity_sleep_windows = _select_activity_sleep_windows(
        sleep_results["sleep_outputs_by_source"],
        participant=participant,
        visit=visit,
        timezone=timezone,
    )
    if activity_sleep_windows.empty:
        reason = (
            "No activity-intensity SPT windows were available after applying the "
            "priority order sleep diary -> HDCZA."
        )
        activity_intensity_results = _skipped_stage_reason("activity_intensity", reason)
        stage_status.append(activity_intensity_results.copy())
    else:
        try:
            activity_intensity_results = run_activity_intensity_pipeline_gp_from_preprocessed(
                calibrated_df=preprocessed["calibrated_df"],
                acc_df=preprocessed["acc_df"],
                temp_df=preprocessed["temp_df"],
                info=preprocessed["info"],
                nonwear_df=preprocessed["nonwear_df"],
                sleep_windows=activity_sleep_windows,
                participant=participant,
                visit=visit,
                epoch_seconds=activity_epoch_seconds,
                thresholds_mg=intensity_thresholds_mg,
                dayborder_hours=dayborder_hours,
                timezone=timezone,
                min_wear_hours=min_wear_hours,
                min_valid_days=min_valid_days,
                max_days=max_days,
                require_spt_overlap=require_activity_spt_overlap,
            )
            stage_status.append({"stage": "activity_intensity", "status": "completed", "reason": ""})

            if save_folder is not None:
                write_activity_intensity_outputs(
                    activity_intensity_results,
                    output_dir=save_folder,
                    include_epoch_table=return_epoch_outputs,
                )
        except ValueError as exc:
            if not _is_min_valid_days_error(exc):
                raise
            activity_intensity_results = _skipped_stage_result("activity_intensity", exc)
            stage_status.append(activity_intensity_results.copy())

    combined_info = dict(preprocessed["info"])
    combined_info["available_guider_sources"] = sleep_results["info"].get("available_guider_sources", [])
    combined_info["activity_sleep_window_sources_used"] = (
        sorted(activity_sleep_windows["sleep_window_source"].unique().tolist())
        if not activity_sleep_windows.empty
        else []
    )
    combined_info["n_activity_sleep_windows"] = int(len(activity_sleep_windows))
    combined_info["stage_status"] = stage_status

    if save_folder is not None:
        pd.DataFrame(stage_status).to_csv(Path(save_folder) / "pipeline_stage_status.csv", index=False)

    return {
        "participant": participant,
        "visit": visit,
        "info": combined_info,
        "stage_status": stage_status,
        "preprocessed": preprocessed,
        "sleep_results": sleep_results,
        "circadian_results": circadian_results,
        "activity_intensity_results": activity_intensity_results,
        "activity_sleep_windows_df": activity_sleep_windows,
    }


def run_sleep_pipeline_gp_from_silver(silver_root: str | Path,
                                      participant: str,
                                      visit: str = "T0",
                                      sensor: str = "Empatica",
                                      **kwargs) -> Dict[str, object]:
    """
    Convenience wrapper to run the GP sleep pipeline from the standard silver layout.
    """
    paths = find_empatica_parquet_files(
        silver_root=silver_root,
        participant=participant,
        visit=visit,
        sensor=sensor,
    )
    return run_sleep_pipeline_gp(
        acc_parquet_path=paths["acc_parquet_path"],
        temp_parquet_path=paths["temp_parquet_path"],
        participant=participant,
        visit=visit,
        **kwargs,
    )


def run_sleep_and_circadian_pipeline_gp_from_silver(silver_root: str | Path,
                                                    participant: str,
                                                    recruitment_tracker_path: Optional[str | Path] = None,
                                                    visit: str = "T0",
                                                    sensor: str = "Empatica",
                                                    **kwargs) -> Dict[str, object]:
    """
    Convenience wrapper to run the combined GP sleep + circadian pipeline from
    the standard silver layout.
    """
    paths = find_empatica_parquet_files(
        silver_root=silver_root,
        participant=participant,
        visit=visit,
        sensor=sensor,
    )
    return run_sleep_and_circadian_pipeline_gp(
        acc_parquet_path=paths["acc_parquet_path"],
        temp_parquet_path=paths["temp_parquet_path"],
        participant=participant,
        visit=visit,
        recruitment_tracker_path=recruitment_tracker_path,
        **kwargs,
    )


__all__ = [
    "run_sleep_pipeline_gp_from_preprocessed",
    "run_sleep_pipeline_gp",
    "run_sleep_pipeline_gp_from_silver",
    "run_sleep_and_circadian_pipeline_gp",
    "run_sleep_and_circadian_pipeline_gp_from_silver",
]

# Neutral public names; historical names remain aliases to the same functions.
run_sleep_from_preprocessed = run_sleep_pipeline_gp_from_preprocessed
run_sleep_from_empatica = run_sleep_pipeline_gp
run_sleep_from_empatica_silver = run_sleep_pipeline_gp_from_silver
run_wrist_from_empatica = run_sleep_and_circadian_pipeline_gp
run_wrist_from_empatica_silver = run_sleep_and_circadian_pipeline_gp_from_silver
__all__ += ["run_sleep_from_preprocessed", "run_sleep_from_empatica", "run_sleep_from_empatica_silver",
            "run_wrist_from_preprocessed", "run_wrist_from_empatica", "run_wrist_from_empatica_silver"]
