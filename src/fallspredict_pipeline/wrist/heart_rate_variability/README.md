# Heart Rate Variability

This module contains the Empatica PPG HRV workflow migrated from `legacy/hrv/HRV.ipynb`.

The algorithm files were adapted from the original notebook:

- `compute_acc_metrics.py`
- `detect_acc_bursts.py`
- `ppg_beat_detection.py`
- `kubios.py`
- `heart_rate_fragmentation.py`

The package-level additions are intentionally thin:

- `config.py`: HRV-specific paths and parameters
- `io.py`: parquet readers for `ppg.parquet` and `acc.parquet`
- `quiet_periods.py`: quiet-period and variable-window helpers from the notebook
- `pipeline.py`: participant and batch orchestration

## Inputs

For each participant/visit, the runner expects:

- `input_root/<participant>/<visit>/Empatica/ppg.parquet`
- `input_root/<participant>/<visit>/Empatica/acc.parquet`
- the sleep pipeline output at `sleep_output_root/<participant>/<visit>/Empatica/sleep_circadian/sleep_output_all_guiders.csv`

The HRV runner segments PPG/ACC using the derived `spt_start` and `spt_end`
from the sleep output. If multiple guiders are available for the same night,
the default priority is:

- `sleep_diary`
- `HDCZA`
- `lower_back_tib`

An older `night_aggregation.csv`-style sleep-window file can still be provided
as `sleep_windows_path`, but it is treated as a fallback only when the sleep
pipeline output is unavailable.

## Outputs

By default, outputs preserve the legacy notebook location:

- `output_root/<participant>/<visit>/Empatica/<output_subdir>/hrv_night.csv`
- `output_root/<participant>/<visit>/Empatica/<output_subdir>/ibi_quiet.parquet`
- `output_root/<participant>/<visit>/Empatica/<output_subdir>/hrv_sleep_windows_selected.csv`

The raw nocturnal accelerometer burst detections used to define quiet periods
are also saved per participant/visit:

- `output_root/<participant>/<visit>/Empatica/nocturnal_bursts/nocturnal_bursts.csv`

## Run

```bash
gp-hrv --config configs/hrv.local.toml --participant 900001
```

The same workflow can be run interactively from:

- `notebooks/run_hrv_pipeline.ipynb`

## Recording gaps and interpolation quality

Processing intersects continuous ACC and PPG coverage within each selected
sleep window. It never runs beat detection or interpolation across a missing
sample. Aware timestamps preserve their instants; naive clock times use the
configured timezone.

Interpolation remains enabled, with no fraction-based exclusion. Window rows
include `contains_interpolated_beats`, `n_interpolated_beats` and
`interpolated_fraction` (filled intervals / retained intervals in the window).
The IBI parquet has `ibi` in seconds and a boolean `interpolated` column.
Intervals use real consecutive peak times and are indexed by the ending peak.
Burst detection preserves timezone information and handles stationary signals.

Only complete nonempty HRV results receive `hrv_completion.json`. Reuse requires
matching parameters, selected sleep windows, sensor input paths/sizes/times and
hashes for all four expected output tables. Empty runs are retryable; a failed
rerun removes partial and previous derived tables. Reprocess old results before
rebuilding cohort aggregations.
