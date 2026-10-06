from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from dare_wearables.common.protocols import LowerBackOptions
from dare_wearables.lower_back.gait.reorientation_functions import build_flipped_mask, detect_orientation_flips
from dare_wearables.lower_back.posture.lying_functions import (
    detect_lying_lowerback,
    detect_naps,
    fill_short_nonwear_between_lying,
    qc_tib_night_wear,
    vanhees_tib_from_bouts,
)
from dare_wearables.lower_back.preprocessing import preprocess_lowerback_file
from dare_wearables.lower_back.utils.bool_to_bouts import bool_to_bouts


MIN_SIZE = 300 * 1024 * 1024

acc_cols = ["acc_x", "acc_y", "acc_z"]
gyro_cols = ["gyr_x", "gyr_y", "gyr_z"]


def _load_gait_helpers():
    try:
        from dare_wearables.lower_back.gait.functions import get_cohort, get_sensor_height, process_gait_data
    except ModuleNotFoundError as exc:
        if exc.name == "mobgap":
            raise ModuleNotFoundError(
                "The gait pipeline requires the `mobgap` package. "
                "Install it in the Python environment used to run this script, "
                "or run lower_back_pipeline.py with --skip-gait to produce posture/time-in-bed outputs."
            ) from exc
        raise
    return get_cohort, get_sensor_height, process_gait_data


def load_redcap_metadata(redcap_csv: str | Path, *, get_sensor_height_fn=None):
    if get_sensor_height_fn is None:
        _, get_sensor_height_fn, _ = _load_gait_helpers()
    redcap_df = pd.read_csv(redcap_csv, dtype={"patient_id": "string"})
    redcap_df.rename(columns={"obs_pt_height": "Height"}, inplace=True)

    sensor_heights = get_sensor_height_fn(redcap_df)
    sensor_heights = sensor_heights[
        sensor_heights["redcap_event_name"].str.contains("baseline_arm_1", case=False, na=False)
    ].copy()
    from dare_wearables.lower_back.gait.functions import normalize_participant_id
    sensor_heights["patient_id"] = sensor_heights["patient_id"].map(normalize_participant_id)
    sensor_heights = sensor_heights[["patient_id", "Height"]].copy()
    return redcap_df, sensor_heights


def apply_orientation_correction(df):
    orientation_windows, segments, flip_times = detect_orientation_flips(
        df,
        cols=acc_cols,
        window="10min",
    )
    flipped_mask = build_flipped_mask(df, segments)

    df_corrected = df.copy()
    df_corrected.loc[flipped_mask, acc_cols] *= -1

    existing_gyro_cols = [c for c in gyro_cols if c in df_corrected.columns]
    df_corrected.loc[flipped_mask, existing_gyro_cols] *= -1
    return df_corrected


def run_gait_pipeline(df, *, subject, cohort, sensor_height_m, save_folder, process_gait_data_fn=None):
    from dare_wearables.lower_back.gait.daily import run_daily_gait
    if process_gait_data_fn is None:
        from dare_wearables.lower_back.gait.functions import process_gait_data
        process_gait_data_fn = process_gait_data
    return run_daily_gait(df, subject=subject, cohort=cohort, sensor_height_m=sensor_height_m,
                          save_folder=save_folder, processor=process_gait_data_fn)


def run_time_in_bed_pipeline(df, *, save_root):
    posture_folder = os.path.join(save_root, "posture")
    nonwear_folder = os.path.join(save_root, "nonwear")
    os.makedirs(posture_folder, exist_ok=True)
    os.makedirs(nonwear_folder, exist_ok=True)

    labels_1hz, theta_1hz, lying_bouts = detect_lying_lowerback(df, out_fs_hz=1)

    wear_1hz = df["wear"].astype(float).resample("1s").mean() >= 0.5
    wear_fixed_1hz, nonwear_bouts, nonwear_filled = fill_short_nonwear_between_lying(
        wear_1hz=wear_1hz,
        lying_1hz=labels_1hz,
        max_nonwear="2h",
        edge_tol="2min",
    )

    labels, theta, lying_bouts = detect_lying_lowerback(
        df,
        out_fs_hz=1,
        wear_1hz_override=wear_fixed_1hz,
    )

    blocks_merged, tib_df = vanhees_tib_from_bouts(
        lying_bouts,
        min_block="30min",
        gap_merge="45min",
        day_offset_hours=12,
    )

    naps_df = detect_naps(
        lying_bouts=lying_bouts,
        tib_df=tib_df,
        min_nap="20min",
        nap_gap_merge="5min",
    )

    tib_qc_df, tib_valid_df = qc_tib_night_wear(
        tib_df=tib_df,
        wear_1hz=wear_fixed_1hz,
        night_start_h=18,
        night_end_h=11,
        min_wear_hours_night=8.0,
        max_cont_nonwear_hours_night=3.0,
        min_wear_frac_in_tib=0.85,
        tib_min_h=3.0,
        tib_max_h=14.0,
        min_overlap_night_h=4.0,
    )

    nw_bouts = bool_to_bouts(~wear_1hz, state=True)

    nw_bouts.to_csv(os.path.join(nonwear_folder, "nonwear_bouts.csv"), index=False)
    lying_bouts.to_csv(os.path.join(posture_folder, "lying_bouts.csv"), index=False)
    blocks_merged.to_csv(os.path.join(posture_folder, "tib_blocks_merged.csv"), index=False)
    tib_df.to_csv(os.path.join(posture_folder, "tib_df.csv"), index=False)
    naps_df.to_csv(os.path.join(posture_folder, "naps_df.csv"), index=False)
    tib_qc_df.to_csv(os.path.join(posture_folder, "tib_qc_df.csv"), index=False)
    tib_valid_df.to_csv(os.path.join(posture_folder, "tib_valid_df.csv"), index=False)
    nonwear_filled.to_csv(os.path.join(nonwear_folder, "nonwear_filled_between_lying.csv"), index=False)


def _list_subjects(
    bronze_root: Path,
    participant_ids: list[str] | None = None,
) -> list[str]:
    if participant_ids:
        return [str(subject) for subject in participant_ids]

    return sorted(
        subject_dir.name
        for subject_dir in bronze_root.iterdir()
        if subject_dir.is_dir()
    )


def run_from_config(config: LowerBackOptions) -> None:
    main(
        bronze_root=config.bronze_root,
        silver_root=config.silver_root,
        redcap_csv=config.redcap_csv,
        visits=config.visits,
        sensor=config.sensor,
        min_size_bytes=config.min_size_bytes,
        participant_ids=config.participant_ids,
        run_gait=config.run_gait,
        run_time_in_bed=config.run_time_in_bed,
        save_omx_to_parquet=config.save_omx_to_parquet,
    )


def main(
    *,
    bronze_root: str | Path,
    silver_root: str | Path,
    redcap_csv: str | Path | None = None,
    visits: list[str] | None = None,
    sensor: str = "McRoberts",
    min_size_bytes: int = MIN_SIZE,
    participant_ids: list[str] | None = None,
    run_gait: bool = True,
    run_time_in_bed: bool = True,
    save_omx_to_parquet: bool = True,
) -> None:
    bronze_root = Path(bronze_root)
    silver_root = Path(silver_root)
    visit_list = visits or ["T0", "T1"]

    redcap_df = sensor_heights = None
    get_cohort = get_sensor_height = process_gait_data = None
    if run_gait:
        if redcap_csv is None:
            raise ValueError("`redcap_csv` is required when `run_gait=True`.")
        get_cohort, get_sensor_height, process_gait_data = _load_gait_helpers()
        redcap_df, sensor_heights = load_redcap_metadata(
            redcap_csv,
            get_sensor_height_fn=get_sensor_height,
        )

    subjects = _list_subjects(bronze_root, participant_ids=participant_ids)
    print(f"Found {len(subjects)} participants:\n{subjects}")

    for subject in subjects:
        print(f"\nProcessing: {subject}")
        for visit in visit_list:
            print(f"--> Visit: {visit} ", end="", flush=True)

            folder_mcroberts = bronze_root / subject / visit / sensor
            if not folder_mcroberts.is_dir():
                print(f"No folder {folder_mcroberts}")
                continue

            omx_files = sorted(path for path in folder_mcroberts.iterdir() if path.suffix == ".OMX")
            if not omx_files:
                print("No data")
                continue

            filepath = omx_files[0]
            if not filepath.exists():
                print(f"File not found: {filepath}")
                continue

            size = filepath.stat().st_size
            if size < min_size_bytes:
                print(f"Skipping {filepath}: {size / (1024 * 1024):.2f} MB (too small)")
                continue

            print(f"Processing {filepath}: {size / (1024 * 1024):.2f} MB")

            df = preprocess_lowerback_file(str(filepath))["df"]
            preprocessed_folder = silver_root / subject / visit / sensor
            preprocessed_folder.mkdir(parents=True, exist_ok=True)
            if save_omx_to_parquet:
                df.to_parquet(preprocessed_folder / f"{subject}_{visit}_preprocessed.parq")

            df = df.set_index("datetime").copy()
            df = apply_orientation_correction(df)
            print(f"Duration: {np.round(len(df) / 100 / 60 / 60 / 24, 2)} days")

            if run_gait:
                sub = subject.lstrip("0").strip()
                sensor_height = get_sensor_height(redcap_df, subject=sub)
                sub_cohort = get_cohort(redcap_df, sub)
                print(f"Sensor height: {sensor_height} m")
                print(f"Cohort: {sub_cohort}")
                run_gait_pipeline(
                    df.copy(),
                    subject=subject,
                    cohort=sub_cohort,
                    sensor_height_m=sensor_height,
                    save_folder=str(preprocessed_folder / "gait"),
                    process_gait_data_fn=process_gait_data,
                )

            if run_time_in_bed:
                run_time_in_bed_pipeline(
                    df.copy(),
                    save_root=str(preprocessed_folder),
                )

run_lowerback_batch = main
