from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from dare_wearables.common.recording import continuous_intersections, in_timezone, timestamp_in_timezone
from dare_wearables.common.output_state import OutputRun, signature, clear_outputs_on_failure

from dare_wearables.wrist.heart_rate_variability.compute_acc_metrics import compute_acc_SMV
from dare_wearables.wrist.heart_rate_variability.config import HRVConfig
from dare_wearables.wrist.heart_rate_variability.detect_acc_bursts import detect_bursts
from dare_wearables.wrist.heart_rate_variability.heart_rate_fragmentation import compute_HRF
from dare_wearables.wrist.heart_rate_variability.io import find_empatica_heart_rate_files, load_acc, load_ppg
from dare_wearables.wrist.heart_rate_variability.kubios import signal_fixpeaks
from dare_wearables.wrist.heart_rate_variability.ppg_beat_detection import MSPTDfast
from dare_wearables.wrist.heart_rate_variability.quiet_periods import build_quiet_periods, build_variable_windows


def build_hrv_output_dir(
    output_root: str | Path,
    *,
    participant: str,
    visit: str,
    sensor: str,
    output_subdir: str = "hrv",
) -> Path:
    return Path(output_root) / str(participant) / str(visit) / str(sensor) / output_subdir


def build_nocturnal_bursts_output_dir(
    output_root: str | Path,
    *,
    participant: str,
    visit: str,
    sensor: str,
    burst_output_subdir: str = "nocturnal_bursts",
) -> Path:
    return Path(output_root) / str(participant) / str(visit) / str(sensor) / burst_output_subdir


def build_sleep_output_path(
    sleep_output_root: str | Path,
    *,
    participant: str,
    visit: str,
    sensor: str,
    sleep_output_subdir: str = "sleep_circadian",
    sleep_output_filename: str = "sleep_output_all_guiders.csv",
) -> Path:
    return (
        Path(sleep_output_root)
        / str(participant)
        / str(visit)
        / str(sensor)
        / sleep_output_subdir
        / sleep_output_filename
    )


def _intervals_overlap(
    start_a: pd.Timestamp,
    end_a: pd.Timestamp,
    start_b: pd.Timestamp,
    end_b: pd.Timestamp,
) -> bool:
    return bool((start_a < end_b) and (end_a > start_b))


def _participant_mask(series: pd.Series, participant: str) -> pd.Series:
    participant = str(participant)
    mask = series.astype(str).eq(participant)
    try:
        mask = mask | pd.to_numeric(series, errors="coerce").eq(int(participant))
    except ValueError:
        pass
    return mask.fillna(False)


def _coerce_spt_found(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y"})


def _select_sleep_nights_from_pipeline_output(
    sleep_output_df: pd.DataFrame,
    *,
    guider_priority: Iterable[str] = ("sleep_diary", "HDCZA", "lower_back_tib"),
    source_col: str = "guider_source",
    start_col: str = "spt_start",
    end_col: str = "spt_end",
    cover_start_col: str = "guider_start",
    cover_end_col: str = "guider_end",
) -> pd.DataFrame:
    required = (source_col, start_col, end_col)
    for column in required:
        if column not in sleep_output_df.columns:
            raise ValueError(f"Sleep pipeline output is missing required column: {column}")

    sleep_df = sleep_output_df.copy()
    if "spt_found" in sleep_df.columns:
        sleep_df = sleep_df.loc[_coerce_spt_found(sleep_df["spt_found"])].copy()

    sleep_df[start_col] = pd.to_datetime(sleep_df[start_col], errors="coerce")
    sleep_df[end_col] = pd.to_datetime(sleep_df[end_col], errors="coerce")
    sleep_df = sleep_df.dropna(subset=[source_col, start_col, end_col])
    sleep_df = sleep_df.loc[sleep_df[end_col] > sleep_df[start_col]].copy()

    if sleep_df.empty:
        return pd.DataFrame(
            columns=[
                "night_id",
                "source_night_id",
                "sleep_window_source",
                "priority_rank",
                "spt_start",
                "spt_end",
                "cover_start",
                "cover_end",
            ]
        )

    if cover_start_col in sleep_df.columns and cover_end_col in sleep_df.columns:
        sleep_df[cover_start_col] = pd.to_datetime(sleep_df[cover_start_col], errors="coerce")
        sleep_df[cover_end_col] = pd.to_datetime(sleep_df[cover_end_col], errors="coerce")
    else:
        sleep_df[cover_start_col] = pd.NaT
        sleep_df[cover_end_col] = pd.NaT

    sleep_df["cover_start"] = sleep_df[cover_start_col].where(
        sleep_df[cover_start_col].notna(),
        sleep_df[start_col],
    )
    sleep_df["cover_end"] = sleep_df[cover_end_col].where(
        sleep_df[cover_end_col].notna(),
        sleep_df[end_col],
    )

    selected_rows: list[dict[str, object]] = []
    selected_cover_windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    for priority_rank, source in enumerate(guider_priority, start=1):
        source_df = sleep_df.loc[sleep_df[source_col].astype(str) == str(source)].copy()
        source_df = source_df.sort_values(["cover_start", start_col]).reset_index(drop=True)
        for row in source_df.itertuples(index=False):
            cover_start = getattr(row, "cover_start")
            cover_end = getattr(row, "cover_end")
            overlaps_selected = any(
                _intervals_overlap(cover_start, cover_end, start, end)
                for start, end in selected_cover_windows
            )
            if overlaps_selected:
                continue

            source_night_id = getattr(row, "night_id", len(selected_rows) + 1)
            selected_rows.append({
                "source_night_id": source_night_id,
                "sleep_window_source": source,
                "priority_rank": priority_rank,
                "spt_start": getattr(row, start_col),
                "spt_end": getattr(row, end_col),
                "cover_start": cover_start,
                "cover_end": cover_end,
            })
            selected_cover_windows.append((cover_start, cover_end))

    if not selected_rows:
        return pd.DataFrame(
            columns=[
                "night_id",
                "source_night_id",
                "sleep_window_source",
                "priority_rank",
                "spt_start",
                "spt_end",
                "cover_start",
                "cover_end",
            ]
        )

    selected = pd.DataFrame(selected_rows)
    selected = selected.sort_values("spt_start").reset_index(drop=True)
    selected["night_id"] = np.arange(1, len(selected) + 1)
    return selected[
        [
            "night_id",
            "source_night_id",
            "sleep_window_source",
            "priority_rank",
            "spt_start",
            "spt_end",
            "cover_start",
            "cover_end",
        ]
    ]


def _select_sleep_nights(
    sleep_windows_df: pd.DataFrame,
    *,
    participant: str,
    subject_col: str = "subject",
    start_col: str = "spt_start_sleeplog",
    end_col: str = "spt_end_sleeplog",
) -> np.ndarray:
    for column in (subject_col, start_col, end_col):
        if column not in sleep_windows_df.columns:
            raise ValueError(f"Sleep windows file is missing required column: {column}")

    sleep_sub = sleep_windows_df.loc[_participant_mask(sleep_windows_df[subject_col], participant)]
    sleep_sub = sleep_sub.dropna(subset=[start_col, end_col])
    return sleep_sub[[start_col, end_col]].values


def _load_selected_sleep_windows(
    *,
    sleep_output_root: str | Path | None,
    sleep_windows_path: str | Path | None,
    output_root: str | Path,
    participant: str,
    visit: str,
    sensor: str,
    sleep_output_subdir: str,
    sleep_output_filename: str,
    sleep_guider_priority: Iterable[str],
    sleep_subject_col: str,
    sleep_start_col: str,
    sleep_end_col: str,
) -> tuple[pd.DataFrame, Path | None, str]:
    sleep_output_path = build_sleep_output_path(
        sleep_output_root or output_root,
        participant=participant,
        visit=visit,
        sensor=sensor,
        sleep_output_subdir=sleep_output_subdir,
        sleep_output_filename=sleep_output_filename,
    )

    if sleep_output_path.exists():
        sleep_output_df = pd.read_csv(sleep_output_path)
        selected = _select_sleep_nights_from_pipeline_output(
            sleep_output_df,
            guider_priority=sleep_guider_priority,
        )
        return selected, sleep_output_path, "sleep_pipeline"

    if sleep_windows_path is not None:
        fallback_path = Path(sleep_windows_path)
        if fallback_path.exists():
            legacy_df = pd.read_csv(fallback_path)
            nights = _select_sleep_nights(
                legacy_df,
                participant=participant,
                subject_col=sleep_subject_col,
                start_col=sleep_start_col,
                end_col=sleep_end_col,
            )
            selected = pd.DataFrame(nights, columns=["spt_start", "spt_end"])
            selected["spt_start"] = pd.to_datetime(selected["spt_start"], errors="coerce")
            selected["spt_end"] = pd.to_datetime(selected["spt_end"], errors="coerce")
            selected = selected.dropna(subset=["spt_start", "spt_end"])
            selected = selected.loc[selected["spt_end"] > selected["spt_start"]].copy()
            selected = selected.sort_values("spt_start").reset_index(drop=True)
            selected["night_id"] = np.arange(1, len(selected) + 1)
            selected["source_night_id"] = selected["night_id"]
            selected["sleep_window_source"] = "legacy_sleep_windows"
            selected["priority_rank"] = 999
            selected["cover_start"] = selected["spt_start"]
            selected["cover_end"] = selected["spt_end"]
            return selected[
                [
                    "night_id",
                    "source_night_id",
                    "sleep_window_source",
                    "priority_rank",
                    "spt_start",
                    "spt_end",
                    "cover_start",
                    "cover_end",
                ]
            ], fallback_path, "legacy_sleep_windows"

    return pd.DataFrame(
        columns=[
            "night_id",
            "source_night_id",
            "sleep_window_source",
            "priority_rank",
            "spt_start",
            "spt_end",
            "cover_start",
            "cover_end",
        ]
    ), sleep_output_path, "missing"


def _empty_ibi_quiet() -> pd.Series:
    return pd.Series(dtype=float)


def _empty_nocturnal_bursts() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "participant",
            "visit",
            "night_id",
            "source_night_id",
            "sleep_window_source",
            "priority_rank",
            "spt_start",
            "spt_end",
            "start",
            "end",
            "duration",
            "peak-to-peak",
            "AUC",
        ]
    )


@clear_outputs_on_failure({
    "output_subdir": ["hrv_night.csv", "ibi_quiet.parquet", "hrv_sleep_windows_selected.csv", "hrv_completion.json"],
    "burst_output_subdir": ["nocturnal_bursts.csv"],
})
def run_hrv_pipeline(
    *,
    input_root: str | Path,
    output_root: str | Path,
    sleep_output_root: str | Path | None = None,
    sleep_windows_path: str | Path | None = None,
    participant: str,
    visit: str = "T0",
    sensor: str = "Empatica",
    timezone: str = "Europe/Rome",
    output_subdir: str = "hrv",
    burst_output_subdir: str = "nocturnal_bursts",
    sleep_output_subdir: str = "sleep_circadian",
    sleep_output_filename: str = "sleep_output_all_guiders.csv",
    sleep_guider_priority: Iterable[str] = ("sleep_diary", "HDCZA", "lower_back_tib"),
    sampling_frequency: int = 64,
    threshold_bursts: float = 35 / 1000,
    min_window: str | pd.Timedelta = "1 min",
    max_window: str | pd.Timedelta = "5 min",
    window_step: str | pd.Timedelta = "1 min",
    min_beats_per_window: int = 30,
    skip_existing: bool = True,
    sleep_subject_col: str = "subject",
    sleep_start_col: str = "spt_start_sleeplog",
    sleep_end_col: str = "spt_end_sleeplog",
) -> dict[str, Any]:
    """Run the Empatica PPG HRV workflow for one participant/visit."""
    sleep_guider_priority = tuple(sleep_guider_priority)
    if sampling_frequency <= 0 or min_beats_per_window < 2:
        raise ValueError("Sampling frequency must be positive and at least two beats are required.")
    parameters = {k: v for k, v in locals().items() if k != "skip_existing"}


    output_dir = build_hrv_output_dir(
        output_root,
        participant=participant,
        visit=visit,
        sensor=sensor,
        output_subdir=output_subdir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    bursts_output_dir = build_nocturnal_bursts_output_dir(
        output_root,
        participant=participant,
        visit=visit,
        sensor=sensor,
        burst_output_subdir=burst_output_subdir,
    )
    bursts_output_dir.mkdir(parents=True, exist_ok=True)

    hrv_output_path = output_dir / "hrv_night.csv"
    ibi_output_path = output_dir / "ibi_quiet.parquet"
    bursts_output_path = bursts_output_dir / "nocturnal_bursts.csv"
    run = OutputRun(output_dir / "hrv_completion.json", [hrv_output_path, ibi_output_path,
                    bursts_output_path, output_dir / "hrv_sleep_windows_selected.csv"])
    files = find_empatica_heart_rate_files(
        input_root=input_root,
        participant=participant,
        visit=visit,
        sensor=sensor,
    )
    if files["ppg"] is None or files["acc"] is None:
        run.clear()
        return {
            "participant": participant,
            "visit": visit,
            "status": "skipped",
            "reason": "Missing ppg.parquet or acc.parquet",
            "files": files,
            "output_dir": output_dir,
        }

    sleep_windows_df, sleep_windows_source_path, sleep_windows_source = _load_selected_sleep_windows(
        sleep_output_root=sleep_output_root,
        sleep_windows_path=sleep_windows_path,
        output_root=output_root,
        participant=participant,
        visit=visit,
        sensor=sensor,
        sleep_output_subdir=sleep_output_subdir,
        sleep_output_filename=sleep_output_filename,
        sleep_guider_priority=sleep_guider_priority,
        sleep_subject_col=sleep_subject_col,
        sleep_start_col=sleep_start_col,
        sleep_end_col=sleep_end_col,
    )
    parameters["selected_sleep_windows"] = sleep_windows_df.to_json(date_format="iso")
    key = signature(parameters, [files["ppg"], files["acc"], sleep_windows_source_path])
    if skip_existing and run.reusable(key):
        return {
            "participant": participant,
            "visit": visit,
            "status": "skipped",
            "reason": "HRV already computed",
            "output_dir": output_dir,
            "hrv_output_path": hrv_output_path,
            "ibi_output_path": ibi_output_path,
            "bursts_output_dir": bursts_output_dir,
            "bursts_output_path": bursts_output_path,
        }

    run.clear()
    if sleep_windows_df.empty:
        return {
            "participant": participant,
            "visit": visit,
            "status": "skipped",
            "reason": "No sleep windows found",
            "sleep_windows_source_path": sleep_windows_source_path,
            "sleep_windows_source": sleep_windows_source,
            "output_dir": output_dir,
        }
    ppg_df = in_timezone(load_ppg(files["ppg"]), timezone)
    acc_df = in_timezone(load_acc(files["acc"]), timezone)
    if "ppg" not in ppg_df or acc_df.shape[1] != 3:
        raise ValueError("HRV requires a ppg column and three accelerometer axes.")

    sleep_windows_df.to_csv(output_dir / "hrv_sleep_windows_selected.csv", index=False)

    min_window = pd.Timedelta(min_window)
    max_window = pd.Timedelta(max_window)
    window_step = pd.Timedelta(window_step)
    if min_window <= pd.Timedelta(0) or max_window < min_window or window_step <= pd.Timedelta(0):
        raise ValueError("HRV window durations must be positive and max_window >= min_window.")

    hrv_rows = []
    ibi_quiet_all = []
    nocturnal_bursts_all = []

    for i, row in enumerate(sleep_windows_df.itertuples(index=False)):
        start_sleep = timestamp_in_timezone(row.spt_start, timezone)
        end_sleep = timestamp_in_timezone(row.spt_end, timezone)

        portions = continuous_intersections(acc_df.loc[start_sleep:end_sleep], ppg_df.loc[start_sleep:end_sleep],
                                            gap_seconds=1.5 / sampling_frequency, acc_frequency=sampling_frequency, ppg_frequency=sampling_frequency)
        for acc_portion, ppg_night in portions:
            start_sleep = max(acc_portion.index[0], ppg_night.index[0])
            end_sleep = min(acc_portion.index[-1], ppg_night.index[-1])
            if end_sleep - start_sleep < min_window:
                continue
            acc_night = compute_acc_SMV(acc_portion)

            if len(acc_night) == 0:
                continue

            bursts = detect_bursts(acc_night, sampling_rate=sampling_frequency, alfa=threshold_bursts)
            raw_bursts = bursts.copy()
            if not raw_bursts.empty:
                raw_bursts.insert(0, "participant", participant)
                raw_bursts.insert(1, "visit", visit)
                raw_bursts.insert(2, "night_id", row.night_id)
                raw_bursts.insert(3, "source_night_id", row.source_night_id)
                raw_bursts.insert(4, "sleep_window_source", row.sleep_window_source)
                raw_bursts.insert(5, "priority_rank", row.priority_rank)
                raw_bursts.insert(6, "spt_start", start_sleep)
                raw_bursts.insert(7, "spt_end", end_sleep)
                nocturnal_bursts_all.append(raw_bursts)

            if not bursts.empty:
                bursts["duration_s"] = (bursts["end"] - bursts["start"]).dt.total_seconds()
                bursts = bursts[bursts["duration_s"] >= 2].reset_index(drop=True)

                bursts["end"] = bursts["end"] + pd.Timedelta("1s")

            quiet_periods = build_quiet_periods(start_sleep, end_sleep, bursts)

            for _, quiet_period in quiet_periods.iterrows():
                qstart = quiet_period["start"]
                qend = quiet_period["end"]
                seg_len = qend - qstart

                if seg_len < min_window:
                    continue

                ppg_quiet = ppg_night.loc[qstart:qend]

                if len(ppg_quiet) < min_window.total_seconds() * sampling_frequency:
                    continue

                _, peaks = MSPTDfast(ppg_quiet["ppg"].values, sampling_rate=sampling_frequency)

                if peaks is None or len(peaks) < 3:
                    continue

                t_peaks = ppg_quiet.index[np.asarray(peaks, dtype=int)]
                ibi = np.diff(t_peaks.to_numpy(dtype="datetime64[ns]").astype(np.int64)).astype(float) / 1e9

                if len(ibi) < 3:
                    continue

                ibi = pd.Series(ibi, index=t_peaks[1:])

                artifacts, _ = signal_fixpeaks(ibi.values, sampling_frequency, iterative=False)

                artifacts_all = np.unique(np.concatenate([
                    artifacts["ectopic"],
                    artifacts["missed"],
                    artifacts["extra"],
                    artifacts["longshort"]
                ])).astype(int)

                valid = artifacts_all[(artifacts_all >= 0) & (artifacts_all < len(ibi))]
                n_artifacts = len(valid)

                if n_artifacts > 0:
                    ibi.iloc[valid] = np.nan

                ibi[(ibi < 0.3) | (ibi > 2.0)] = np.nan

                interpolated = ibi.isna()
                ibi_clean = ibi.interpolate(method="linear", limit_direction="both")

                windows = build_variable_windows(
                    qstart, qend,
                    min_window=min_window,
                    max_window=max_window,
                    step=window_step
                )

                for wstart, wend in windows:
                    ibi_window = ibi_clean.loc[wstart:wend].dropna()

                    if len(ibi_window) < min_beats_per_window:
                        continue

                    ppi = ibi_window.values * 1000
                    diff_ppi = np.diff(ppi)

                    if len(diff_ppi) == 0:
                        continue

                    mean_hr = np.mean(60000 / ppi)
                    rmssd = np.sqrt(np.mean(diff_ppi ** 2))
                    sdnn = np.std(ppi, ddof=1) if len(ppi) > 1 else np.nan
                    pip = compute_HRF(ppi)

                    hrv_rows.append({
                        "day": i + 1,
                        "sleep_window_source": row.sleep_window_source,
                        "source_night_id": row.source_night_id,
                        "priority_rank": row.priority_rank,
                        "window_start": wstart,
                        "window_end": wend,
                        "time": wstart + (wend - wstart) / 2,
                        "window_length_s": (wend - wstart).total_seconds(),
                        "quiet_segment_length_s": seg_len.total_seconds(),
                        "n_beats": len(ppi),
                        "n_artifacts_segment": n_artifacts,
                        "n_interpolated_beats": int(interpolated.reindex(ibi_window.index).sum()),
                        "interpolated_fraction": float(interpolated.reindex(ibi_window.index).mean()),
                        "contains_interpolated_beats": bool(interpolated.reindex(ibi_window.index).any()),
                        "mean_hr": mean_hr,
                        "rmssd": rmssd,
                        "sdnn": sdnn,
                        "PIP": pip
                    })

                ibi_quiet_all.append(pd.DataFrame({"ibi": ibi_clean, "interpolated": interpolated}))

    hrv_columns = ["day", "sleep_window_source", "source_night_id", "priority_rank", "window_start", "window_end",
                   "time", "window_length_s", "quiet_segment_length_s", "n_beats", "n_artifacts_segment",
                   "n_interpolated_beats", "interpolated_fraction", "contains_interpolated_beats", "mean_hr", "rmssd", "sdnn", "PIP"]
    hrv_df = pd.DataFrame(hrv_rows, columns=hrv_columns)
    ibi_quiet_df = pd.concat(ibi_quiet_all) if len(ibi_quiet_all) > 0 else pd.DataFrame(columns=["ibi", "interpolated"])
    nocturnal_bursts_df = (
        pd.concat(nocturnal_bursts_all, ignore_index=True, sort=False)
        if nocturnal_bursts_all
        else _empty_nocturnal_bursts()
    )

    hrv_df.to_csv(hrv_output_path)
    ibi_quiet_df.to_parquet(ibi_output_path)
    nocturnal_bursts_df.to_csv(bursts_output_path, index=False)
    if not hrv_df.empty:
        run.complete(key)

    return {
        "participant": participant,
        "visit": visit,
        "status": "completed" if not hrv_df.empty else "skipped",
        "output_dir": output_dir,
        "hrv_output_path": hrv_output_path,
        "ibi_output_path": ibi_output_path,
        "bursts_output_dir": bursts_output_dir,
        "bursts_output_path": bursts_output_path,
        "sleep_windows_output_path": output_dir / "hrv_sleep_windows_selected.csv",
        "sleep_windows_source_path": sleep_windows_source_path,
        "sleep_windows_source": sleep_windows_source,
        "sleep_guider_priority": list(sleep_guider_priority),
        "n_sleep_windows": len(sleep_windows_df),
        "n_nocturnal_bursts": len(nocturnal_bursts_df),
        "n_hrv_windows": len(hrv_df),
        "n_ibi_samples": len(ibi_quiet_df),
    }


def run_hrv_pipeline_from_config(
    config: HRVConfig,
    *,
    participant: str | None = None,
    visit: str | None = None,
) -> dict[str, Any]:
    participant_id = participant or config.participant
    if participant_id is None:
        raise ValueError("A participant must be provided via the config or function call.")

    return run_hrv_pipeline(
        input_root=config.input_root,
        output_root=config.output_root,
        sleep_output_root=config.sleep_output_root,
        sleep_windows_path=config.sleep_windows_path,
        participant=participant_id,
        visit=visit or config.visit,
        sensor=config.sensor,
        timezone=config.timezone,
        output_subdir=config.output_subdir,
        burst_output_subdir=config.burst_output_subdir,
        sleep_output_subdir=config.sleep_output_subdir,
        sleep_output_filename=config.sleep_output_filename,
        sleep_guider_priority=config.sleep_guider_priority,
        sampling_frequency=config.sampling_frequency,
        threshold_bursts=config.threshold_bursts,
        min_window=config.min_window,
        max_window=config.max_window,
        window_step=config.window_step,
        min_beats_per_window=config.min_beats_per_window,
        skip_existing=config.skip_existing,
    )


def run_hrv_batch_from_config(
    config: HRVConfig,
    *,
    participants: Iterable[str] | None = None,
    visit: str | None = None,
) -> list[dict[str, Any]]:
    participant_ids = list(participants) if participants is not None else None
    if participant_ids is None:
        participant_ids = [p.name for p in Path(config.input_root).iterdir() if not p.name.startswith(".")]
        participant_ids = sorted(participant_ids)

    return [
        run_hrv_pipeline_from_config(config, participant=str(participant), visit=visit)
        for participant in participant_ids
    ]
