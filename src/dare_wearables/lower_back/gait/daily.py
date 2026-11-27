"""Daily gait processing that respects wear and recording boundaries."""
from pathlib import Path
import numpy as np
import pandas as pd

WB_COLUMNS = ["start", "end", "duration_s", "n_strides", "stride_duration_s",
              "cadence_spm", "stride_length_m", "walking_speed_mps", "n_raw_initial_contacts", "n_turns"]


def aggregate_bouts(bouts, *, cohort, sensor_height_m, participant_height_m=None):
    from mobgap.aggregation import MobilisedAggregator, apply_thresholds, get_mobilised_dmo_thresholds
    mask = apply_thresholds(bouts, get_mobilised_dmo_thresholds(), cohort=cohort,
                            height_m=participant_height_m, measurement_condition="free_living")
    result = MobilisedAggregator(groupby=None).aggregate(bouts, wb_dmos_mask=mask).aggregated_data_
    if bouts.empty:
        result = pd.DataFrame([{c: (0.0 if c.endswith(("__count", "__sum")) or c == "total_walking_duration_min" else np.nan)
                                for c in result.columns}])
    return result


def run_daily_gait(df, *, subject, cohort, sensor_height_m, save_folder, processor,
                   participant_height_m=None):
    folder = Path(save_folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = {name: folder / f"{subject}_{name}.csv" for name in
             ("per_wb", "day_aggregation", "walking_bouts_datetime", "orientation_qc", "gait_status")}
    # Invalidate the preceding run even if validation or an algorithm fails.
    for path in paths.values():
        path.unlink(missing_ok=True)
    statuses, daily, all_bouts, dated_bouts, orientations = [], [], [], [], []
    try:
        if not isinstance(df.index, pd.DatetimeIndex) or df.index.hasnans or df.index.has_duplicates or not df.index.is_monotonic_increasing:
            raise ValueError("Gait input requires unique, increasing datetime timestamps.")
        if "wear" not in df or df.wear.isna().any() or not df.wear.isin([True, False]).all():
            raise ValueError("Gait input requires a complete boolean wear mask.")
        for day_number, (date, day) in enumerate(df.groupby(df.index.date), start=1):
            # The lower-back protocol is 100 Hz; reject mismatched rates instead of
            # reporting incorrect wear hours. Gaps form separate processing segments.
            delta = day.index.to_series().diff().dt.total_seconds()
            regular = delta.dropna().loc[lambda x: x < 0.015]
            if not regular.empty and not np.allclose(regular, 0.01, rtol=0.02, atol=0.00005):
                raise ValueError("Lower-back gait requires 100 Hz samples.")
            breaks = (~day.wear.astype(bool) | ~day.wear.shift(fill_value=False).astype(bool) |
                      ~np.isclose(delta, 0.01, rtol=0.02, atol=0.00005))
            if "time_segment" in day:
                breaks |= day.time_segment.ne(day.time_segment.shift())
            groups = breaks.cumsum()
            bouts, processed_samples = [], 0
            for _, segment in day.loc[day.wear.astype(bool)].groupby(groups):
                if len(segment) < 100:
                    continue
                processed_samples += len(segment)
                data = segment.reset_index().copy()
                data["timestamp"] = (segment.index - segment.index[0]).total_seconds()
                height_kwargs = {"participant_height_m": participant_height_m} if participant_height_m is not None else {}
                result = processor(data, cohort=cohort, sensor_height_m=sensor_height_m,
                                   measurement_condition="free_living", **height_kwargs)
                if result is None:
                    raise ValueError("Gait processor returned no result for an eligible segment.")
                orientation = result.get("orientation")
                if orientation is not None and not orientation.empty:
                    orientation = orientation.copy()
                    orientation["start_time"] = segment.index[orientation.start.to_numpy(dtype=int)]
                    orientation["end_time"] = segment.index[orientation.end.to_numpy(dtype=int)-1] + pd.Timedelta(milliseconds=10)
                    orientation["day"] = day_number
                    orientations.append(orientation)
                wb = result.get("per_wb_params")
                if wb is None or wb.empty:
                    continue
                wb = wb.copy()
                wb["step_count"] = result["step_counts"]
                offset = day.index.get_indexer([segment.index[0]])[0]
                for row in wb.itertuples():
                    start, end = int(row.start), int(row.end)
                    if not 0 <= start < end <= len(segment):
                        raise ValueError("Walking-bout bounds exceed their continuous segment.")
                    dated_bouts.append({"start": segment.index[start],
                                        "end": segment.index[end-1] + pd.Timedelta(milliseconds=10),
                                        "day": day_number})
                wb[["start", "end"]] += offset
                bouts.append(wb)
            combined = pd.concat(bouts, ignore_index=True) if bouts else pd.DataFrame(columns=WB_COLUMNS + ["step_count"]).astype(float)
            combined.index.name = "wb_id"
            summary = aggregate_bouts(combined.drop(columns="step_count"), cohort=cohort,
                                      sensor_height_m=sensor_height_m, participant_height_m=participant_height_m)
            summary["step_count"] = combined.step_count.sum()
            status = "completed" if processed_samples else "insufficient_continuous_wear"
            if not processed_samples:
                summary.loc[:, :] = np.nan
            summary["ID"], summary["day"], summary["date"] = subject, day_number, str(date)
            summary["processing_status"] = status
            summary["hours"] = len(day) / 100 / 3600
            summary["nonwear_time_minutes"] = (~day.wear.astype(bool)).sum() / 100 / 60
            summary["nonwear_time_percent"] = 100 * (~day.wear.astype(bool)).mean()
            summary["unprocessed_wear_seconds"] = (day.wear.sum() - processed_samples) / 100
            daily.append(summary)
            combined["day"] = day_number
            all_bouts.append(combined)
            statuses.append({"day": day_number, "date": str(date), "status": status, "reason": ""})
        wb_table = pd.concat(all_bouts, ignore_index=True) if all_bouts else pd.DataFrame(columns=WB_COLUMNS + ["step_count", "day"])
        day_table = pd.concat(daily, ignore_index=True) if daily else pd.DataFrame(columns=["ID", "day", "hours", "nonwear_time_minutes"])
        wb_table.to_csv(paths["per_wb"], index=False)
        day_table.to_csv(paths["day_aggregation"], index=False)
        pd.DataFrame(dated_bouts, columns=["start", "end", "day"]).to_csv(paths["walking_bouts_datetime"], index=False)
        orientation_table = (pd.concat(orientations, ignore_index=True) if orientations else
                             pd.DataFrame(columns=["gs_id", "start_time", "end_time", "day", "status", "rotation_deg"]))
        orientation_table.to_csv(paths["orientation_qc"], index=False)
    except Exception as exc:
        for name, path in paths.items():
            if name != "gait_status":
                path.unlink(missing_ok=True)
        statuses.append({"day": None, "date": "", "status": "failed", "reason": str(exc)})
        raise
    finally:
        pd.DataFrame(statuses, columns=["day", "date", "status", "reason"]).to_csv(paths["gait_status"], index=False)
