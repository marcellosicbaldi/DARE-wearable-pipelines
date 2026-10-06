import os
import numpy as np
import pandas as pd
import struct
import glob
from pathlib import Path
from datetime import datetime, timedelta
from typing import Final
from types import MappingProxyType
from mobgap.aggregation import apply_thresholds, get_mobilised_dmo_thresholds, MobilisedAggregator
from mobgap.wba import StrideSelection, WbAssembly
from mobgap.laterality import strides_list_from_ic_lr_list
from mobgap.utils.df_operations import create_multi_groupby
from mobgap.utils.interpolation import naive_sec_paras_to_regions
from mobgap.pipeline import GsIterator
from mobgap.gait_sequences import GsdIonescu, GsdIluz
from mobgap.initial_contacts import IcdShinImproved, IcdIonescu, IcdHKLeeImproved, refine_gs
from mobgap.cadence import CadFromIcDetector
from mobgap.stride_length import SlZijlstra
from mobgap.walking_speed import WsNaive
from mobgap.turning import TdElGohary
from mobgap.utils.conversions import to_body_frame
from mobgap.laterality import LrcUllrich
from mobgap.consts import GRAV_MS2

from dare_wearables.lower_back.preprocessing import preprocess_lowerback_file

# ================
# Define functions
# ================
def process_gait_data(df, cohort="HA", sensor_height_m=1.0, measurement_condition="free_living"):
    """
    Process gait data using the appropriate pipeline based on cohort.
 
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing: timestamp, acceleration, gyroscope data.
    cohort : str
        Patient cohort (e.g., "HA", "PD", "MS")
    sensor_height_m : float
        Subject sensor height in meters, used for stride length calculation.
    measurement_condition : str
        Measurement condition (e.g., "free_living")
    Adapted by Jose AS on 09.06.2025
 
    Returns:
    --------
    dict
        Dictionary containing processed gait parameters and results
    """
    # Define pipelines with their configurations
    regular_walking: Final = MappingProxyType({
        "gait_sequence_detection": GsdIonescu(), # Changed from GsdIluz() to GsdIonescu() because orientation-independent(),
        "initial_contact_detection": IcdIonescu(),
        "laterality_classification": LrcUllrich(**LrcUllrich.PredefinedParameters.msproject_all),
        "cadence_calculation": CadFromIcDetector(IcdShinImproved(), silence_ic_warning=True),
        "stride_length_calculation": SlZijlstra(),
        "walking_speed_calculation": WsNaive(),
        "turn_detection": TdElGohary(),
        "stride_selection": StrideSelection(),
        "wba": WbAssembly(),
        "dmo_thresholds": get_mobilised_dmo_thresholds(),
        "dmo_aggregation": MobilisedAggregator(groupby=None),
        "recommended_cohorts": ("HA", "COPD", "CHF"),
    })
    impaired_walking: Final = MappingProxyType({
        "gait_sequence_detection": GsdIonescu(),
        "initial_contact_detection": IcdIonescu(),
        "laterality_classification": LrcUllrich(**LrcUllrich.PredefinedParameters.msproject_all),
        "cadence_calculation": CadFromIcDetector(IcdHKLeeImproved(), silence_ic_warning=True),
        "stride_length_calculation": SlZijlstra(),
        "walking_speed_calculation": WsNaive(),
        "turn_detection": TdElGohary(),
        "stride_selection": StrideSelection(),
        "wba": WbAssembly(),
        "dmo_thresholds": get_mobilised_dmo_thresholds(),
        "dmo_aggregation": MobilisedAggregator(groupby=None),
        "recommended_cohorts": ("PD", "MS", "PFF"),
    })
 
    # Select appropriate pipeline based on cohort
    if cohort in regular_walking["recommended_cohorts"]:
        pipeline = regular_walking
        #print(f"Using regular walking pipeline for cohort '{cohort}'.")
    elif cohort in impaired_walking["recommended_cohorts"]:
        pipeline = impaired_walking
        print(f"Using impaired walking pipeline for cohort '{cohort}'.")
    else:
        raise ValueError(f"Unsupported gait cohort: {cohort}")

    # Calculate sampling rate
    if len(df['timestamp'].values) < 100: return
    ts = df['timestamp'].values
    sampling_rate_hz = round(1 / np.median(np.diff(ts)))
    if sampling_rate_hz <= 0 or not np.allclose(np.diff(ts), 1 / sampling_rate_hz, rtol=0.02, atol=0.00005):
        raise ValueError("Gait input must be a continuous, regularly sampled segment.")
 
    # Prepare IMU data
    imu_data = to_body_frame(df)
 
    # Use selected pipeline components
    gsd = pipeline["gait_sequence_detection"]
    icd = pipeline["initial_contact_detection"]
    lrc = pipeline["laterality_classification"]
    cad = pipeline["cadence_calculation"]
    sl = pipeline["stride_length_calculation"]
    speed = pipeline["walking_speed_calculation"]
    turn = pipeline["turn_detection"]
    ss = pipeline["stride_selection"]
    wba = pipeline["wba"]
 
    # Gait sequence detection
    gsd.detect(data=imu_data, sampling_rate_hz=sampling_rate_hz)
    gait_sequences = gsd.gs_list_

    if gait_sequences.empty:
            return {
        "aggregated_results": None,
        "per_wb_params": None,
        "final_strides": None,
        "stride_list": None
    }
 
    # Process gait sequences
    gs_iterator = GsIterator()
    for (_, gs_data), r in gs_iterator.iterate(imu_data, gait_sequences):
        icd = icd.clone().detect(gs_data, sampling_rate_hz=sampling_rate_hz)
        lrc = lrc.clone().predict(gs_data, icd.ic_list_, sampling_rate_hz=sampling_rate_hz)
        r.ic_list = lrc.ic_lr_list_
        turn = turn.clone().detect(gs_data, sampling_rate_hz=sampling_rate_hz)
        r.turn_list = turn.turn_list_
        refined_gs, refined_ic_list = refine_gs(r.ic_list)
 
        with gs_iterator.subregion(refined_gs) as ((_, refined_gs_data), rr):
            cad = cad.clone().calculate(
                refined_gs_data,
                initial_contacts=refined_ic_list,
                sampling_rate_hz=sampling_rate_hz
            )
            rr.cadence_per_sec = cad.cadence_per_sec_
 
            sl = sl.clone().calculate(
                refined_gs_data,
                initial_contacts=refined_ic_list,
                sampling_rate_hz=sampling_rate_hz,
                sensor_height_m=sensor_height_m
            )
            rr.stride_length_per_sec = sl.stride_length_per_sec_
 
            speed = speed.clone().calculate(
                refined_gs_data,
                initial_contacts=refined_ic_list,
                cadence_per_sec=cad.cadence_per_sec_,
                stride_length_per_sec=sl.stride_length_per_sec_,
                sampling_rate_hz=sampling_rate_hz
            )
            rr.walking_speed_per_sec = speed.walking_speed_per_sec_
 
    # Process results
    results = gs_iterator.results_
 
    # Combine results
    combined_results = pd.concat(
        [
            results.cadence_per_sec,
            results.stride_length_per_sec,
            results.walking_speed_per_sec,
        ],
        axis=1,
    )
    
    # Get stride list
    stride_list = (
        results.ic_list.groupby("gs_id", group_keys=False)
        .apply(strides_list_from_ic_lr_list)
        .assign(
            stride_duration_s=lambda df_: (df_.end - df_.start) / sampling_rate_hz
        )
    )
 
    # Process stride parameters
    stride_list_with_approx_paras = create_multi_groupby(
        stride_list,
        combined_results,
        "gs_id",
        group_keys=False,
    ).apply(
        safe_naive_sec_paras_to_regions,
        sampling_rate_hz=sampling_rate_hz,
    )

    if stride_list_with_approx_paras.empty:
        return {
            "aggregated_results": None,
            "per_wb_params": None,
            "final_strides": None,
            "stride_list": stride_list,
            "step_counts": None,
        }
    
    # Create flat index
    flat_index = pd.Index(
        ["_".join(str(e) for e in s_id) for s_id in stride_list_with_approx_paras.index],
        name="s_id",
    )
    stride_list_with_approx_paras = (
        stride_list_with_approx_paras.reset_index("gs_id")
        .rename(columns={"gs_id": "original_gs_id"})
        .set_index(flat_index)
    )
 
    # Apply stride selection and walking bout assembly
    ss = ss.filter(stride_list_with_approx_paras, sampling_rate_hz=sampling_rate_hz)
    wba = wba.assemble(ss.filtered_stride_list_, sampling_rate_hz=sampling_rate_hz)
 
    # Get final strides and parameters
    final_strides = wba.annotated_stride_list_
    per_wb_params = wba.wb_meta_parameters_
 
    # Extend with per-stride parameters
    params_to_aggregate = [
        "n_raw_initial_contacts",
        "n_turns",
        "stride_duration_s",
        "cadence_spm",
        "stride_length_m",
        "walking_speed_mps",
    ]
    per_wb_params = pd.concat(
        [
            per_wb_params,
            final_strides.reindex(columns=params_to_aggregate)
            .groupby(["wb_id"])
            .mean(),
        ],
        axis=1,
    )
    step_counts = final_strides.groupby(level=0).size()

    # Apply thresholds
    thresholds = pipeline["dmo_thresholds"]
    per_wb_params_mask = apply_thresholds(
        per_wb_params,
        thresholds,
        cohort=cohort,
        height_m=sensor_height_m,
        measurement_condition=measurement_condition,
    )
 
    # Aggregate results
    agg = pipeline["dmo_aggregation"]
    agg_results = agg.aggregate(
        per_wb_params, wb_dmos_mask=per_wb_params_mask
    ).aggregated_data_
 
    return {
        "aggregated_results": agg_results,
        "per_wb_params": per_wb_params,
        "final_strides": final_strides,
        "stride_list": stride_list,
        "step_counts": step_counts
    }
 
def save_results_mobi(res, save_dir, format='csv'):
    """
    Save results to a file in the specified format.
    :param res: results dictionary
    :param save_dir: directory to save the file
    :param format: format to save the file, options are 'csv', 'parquet', 'hdf', 'feather', 'csv_compressed'
    :return:
    """
    # Process data as before
    keys_list = list(res.keys())
    for key in list(keys_list):
        if not res[key]:
            del res[key]
 
    if not res:
        print("No data to save")
        return
 
    last_key = list(res.keys())[-2]
    df = pd.DataFrame()
 
    for data_key in res[last_key].keys():
        data = res[last_key][data_key]
        if data_key == 'accel' and isinstance(data, np.ndarray) and data.ndim > 1:
            df['acc_x'] = data[:, 0]
            df['acc_y'] = data[:, 1]
            df['acc_z'] = data[:, 2]
        elif data_key == 'gyro' and isinstance(data, np.ndarray) and data.ndim > 1:
            df['gyr_x'] = data[:, 0]
            df['gyr_y'] = data[:, 1]
            df['gyr_z'] = data[:, 2]
        else:
            df[data_key] = data
 
    # Select output format
    if format == 'parquet':
        # Update file extension
        save_path = os.path.splitext(save_dir)[0] + '.parquet'
        df.to_parquet(save_path, compression='snappy')
    elif format == 'hdf':
        save_path = os.path.splitext(save_dir)[0] + '.h5'
        df.to_hdf(save_path, key='data', complevel=9, complib='blosc')
    elif format == 'feather':
        save_path = os.path.splitext(save_dir)[0] + '.feather'
        df.to_feather(save_path)
    elif format == 'csv_compressed':
        save_path = os.path.splitext(save_dir)[0] + '.csv.gz'
        df.to_csv(save_path, index=False, compression='gzip')
    else:  # Default to csv
        df.to_csv(save_dir, index=False)


def convert_time(time):
    new = []
    for datenum in time:
        days = int(datenum)
        fraction = datenum - days
        base_date = datetime(1, 1, 1) + timedelta(days=days - 367)
        time_part = timedelta(days=fraction)
 
        # Final datetime result
        final_datetime = base_date + time_part
        new.append(final_datetime)
    return pd.to_datetime(new)
 
 
def process_fallspredict_files(f):
    return preprocess_lowerback_file(f)["df"]
 
def get_sensor_height(df_clin, method="mobilise", subject=None):
    """Derive height from explicit metadata; never substitute an arbitrary metre."""
    if method.lower() not in ("mobilise", "dempster"):
        raise ValueError(f"Unknown sensor-height method: {method}")
    rows = _participant_baseline(df_clin, subject) if subject is not None else df_clin.copy()
    if "Height" not in rows:
        raise ValueError("Clinical metadata requires Height.")
    rows = rows.dropna(subset=["Height"]).copy()
    heights = pd.to_numeric(rows.Height, errors="raise")
    heights = heights.where(heights <= 10, heights / 100)
    if not np.isfinite(heights).all() or (heights <= 0).any():
        raise ValueError("Height must be positive and finite.")
    rows["Height"] = heights
    rows["sensor_height"] = 0.491 * heights + 0.2 if method.lower() == "mobilise" else 0.53 * heights
    if subject is None:
        return rows
    values = rows.sensor_height.unique()
    if len(values) != 1:
        raise ValueError(f"Missing or conflicting baseline height for {subject}.")
    return float(values[0])

def normalize_participant_id(value):
    """Normalize integer REDCap identifiers without truncation or float rounding."""
    from decimal import Decimal, InvalidOperation
    try:
        number = Decimal(str(value).strip())
    except InvalidOperation as exc:
        raise ValueError(f"Invalid participant identifier: {value!r}") from exc
    if not number.is_finite() or number != number.to_integral_value() or number < 0:
        raise ValueError(f"Invalid participant identifier: {value!r}")
    return str(int(number))


def _participant_baseline(df, subject):
    if "patient_id" not in df:
        raise ValueError("Clinical metadata requires a patient_id column.")
    rows = df.loc[df.patient_id.notna()].copy()
    rows = rows.loc[rows.patient_id.map(normalize_participant_id).eq(normalize_participant_id(subject))]
    if "redcap_event_name" in rows:
        rows = rows.loc[rows.redcap_event_name.astype(str).str.lower().eq("baseline_arm_1")]
    return rows


def get_cohort(df_result, subject):
    """Select the exact participant's baseline diagnosis; ambiguous data is an error."""
    rows = _participant_baseline(df_result, subject)
    values = pd.to_numeric(rows["parkinsonf_b"].dropna(), errors="raise").unique()
    if len(values) != 1 or values[0] not in (0, 1):
        raise ValueError(f"Missing, invalid or conflicting baseline Parkinson status for {subject}.")
    return "PD" if values[0] == 1 else "HA"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()  # Ensure it's written immediately

    def flush(self):
        for s in self.streams:
            s.flush()


# Keep the historical import entry point consistent with preprocessing.
from dare_wearables.lower_back.io.mcroberts_loader import (
    read_dp7_as_dataframe_fast, decode_dp7_timestamp_vector,
    effective_sample_rate, packet_sample_times_vector,
)

def _index_to_numeric_array(index: pd.Index) -> np.ndarray:
    """
    Convert a regular Index or the last level of a MultiIndex to float values.
    MobGap per-second parameter outputs are usually indexed by sample/time values.
    """
    if isinstance(index, pd.MultiIndex):
        values = index.get_level_values(-1)
    else:
        values = index

    return values.to_numpy(dtype=float)


def safe_naive_sec_paras_to_regions(
    regions: pd.DataFrame,
    sec_paras: pd.DataFrame,
    *,
    sampling_rate_hz: float,
) -> pd.DataFrame:
    """
    Safe wrapper around MobGap's naive_sec_paras_to_regions.

    It drops stride/region rows whose start/end are outside the available
    per-second parameter range. This avoids interpolation errors such as:

        A value in x_new is below the interpolation range's minimum value.
    """
    if regions.empty or sec_paras.empty:
        return pd.DataFrame()

    x = _index_to_numeric_array(sec_paras.index)

    x = x[np.isfinite(x)]
    if x.size == 0:
        return pd.DataFrame()

    x_min = float(np.min(x))
    x_max = float(np.max(x))

    regions_safe = regions.copy()

    keep = (
        regions_safe["start"].astype(float).ge(x_min)
        & regions_safe["end"].astype(float).le(x_max)
    )

    regions_safe = regions_safe.loc[keep].copy()

    if regions_safe.empty:
        return pd.DataFrame()

    return naive_sec_paras_to_regions(
        regions_safe,
        sec_paras,
        sampling_rate_hz=sampling_rate_hz,
    )
