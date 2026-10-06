# Aggregation Thresholds and Nonwear Handling

This note summarizes the default GP pipeline rules for deciding whether data are usable for gait, sleep, physical activity, cardio, and circadian outputs, and how nonwear or invalid data are handled.

Scope: the `src/gp_pipeline` package and the current config defaults in `configs/`.
The final cohort minimum-data rule was updated on 2026-10-02.

The final `gp-aggregate-cohorts` step requires at least **3 valid days/nights**
for gait, each sleep guider, HR, HRV and activity intensity. It masks the
corresponding measurements below that minimum and preserves QC counts.
HR and HRV aggregation now export independent coverage counts and also apply
the minimum themselves. Circadian processing already requires three valid days
upstream. See [the count definitions and rerun instructions](combined_cohorts.md#minimum-three-valid-days-or-nights).
The table below distinguishes those final masks from the daily/window rules.

## Quick Reference

| Domain | Default minimum data rule | Where the rule is applied | Subject-level aggregation behavior |
| --- | --- | --- | --- |
| Gait | A day is valid for amount features when `valid_hours > 16.0`, where `valid_hours = hours - nonwear_time_minutes / 60`. | `gp-aggregate-gait` / `aggregate_gait()`; final mask in `gp-aggregate-cohorts` | Amount features average valid days; quality averages all available gait days. Final analysis masks all wearable gait measurements if fewer than 3 valid days. |
| Sleep | Eligible rows have a recognized guider and, when supplied, `spt_found = true`. | `gp-aggregate-sleep` / `aggregate_sleep()`; final mask in `gp-aggregate-cohorts` | Mean/median sleep metrics are masked in final analysis when the corresponding guider has fewer than 3 eligible nights. |
| Physical activity / activity intensity | A day is valid when wear time is at least `18.0` hours. At least `3` valid days are required by default, and at most the first `7` valid days are retained. | Activity-intensity pipeline before `activity_intensity_summary.csv` is written | The aggregation module reads the pipeline summary and applies no additional hour threshold. |
| Circadian | Same day rule as activity: at least `18.0` wear hours per day, at least `3` valid days by default, at most the first `7` valid days. | Circadian pipeline before `circadian_metrics.csv` is written | The aggregation module appends the first row of `circadian_metrics.csv`; it applies no additional hour threshold. |
| Cardio - BeliefPPG heart rate | No additional daily hours threshold. Processing splits at gaps in either sensor, keeps common continuous portions of at least `10 min`, and requires at least `5 min` of samples in each stream for inference. | HR processing and aggregation; final cohort mask | Day HR, night HR and dip each require at least 3 nonmissing day, night or paired estimates respectively. |
| Cardio - HRV | Selected sleep windows, quiet periods at least `1 min`, `1-5 min` windows with a `1 min` step, and at least `30` beats per window. | HRV processing and aggregation; final cohort mask | Medians across windows per night, then across nights. Each physiological metric requires at least 3 nights with retained data after filtering. |

Most thresholds above are defaults in function signatures or CLI arguments, not hard-coded study constants. Gait's `--min-valid-hours` can be changed at aggregation time. Activity and circadian defaults can be changed when calling the wrist pipeline.

## Domain Details

### Gait

The lower-back gait daily files include `hours` and `nonwear_time_minutes`. Subject-level gait aggregation computes:

```text
valid_hours = hours - nonwear_time_minutes / 60
valid_day = valid_hours > min_valid_hours
```

Bologna also has legacy daily files (including the July 2026 exports) with
**no nonwear columns**, produced after nonwear samples had already been
removed. For that format, `hours` is the retained wear duration and
`valid_hours = hours`. The Bologna loader distinguishes these formats per file
before concatenation. An absent nonwear column is not the same as missing
values inside an existing nonwear column: the latter cause an explicit error,
as do missing/nonnumeric hours. Ravenna requires explicit nonwear minutes.
The daily diagnostic `valid_hours_source` records the interpretation and is
not exported as a participant feature. No zero nonwear values are imputed.

This fixes the October refresh's false all-zero Bologna counts: concatenation
previously introduced NaN nonwear values for legacy files, and `NaN > 16`
silently evaluated to False. Private diagnostic scripts and results are not distributed.

The default `min_valid_hours` is `16.0`, and the comparison is strictly greater than 16 hours.

New daily exports also require `processing_status = completed` for a valid
amount day. Gait runs only on continuous wear segments of at least 100 samples
at the lower-back protocol's 100 Hz. Nonwear and timestamp gaps end a segment.
All segment bouts are combined before daily MobGap aggregation. A processed
day without bouts has zero amount features and undefined gait quality; a day
without eligible wear has undefined amounts and
`processing_status = insufficient_continuous_wear`. The daily table reports
`unprocessed_wear_seconds` for wear fragments shorter than the segment minimum.
That field is diagnostic and is excluded from participant feature averages.

Aggregation then separates features into two groups:

- Amount features: `total_walking_duration_min`, `step_count`, and `wb_*__count` / `wb_*__sum` features. These are averaged only over valid days.
- Quality and other gait features: averaged over all available gait days, regardless of valid-day status.

The gait export also includes QC fields such as `gait_n_days`, `gait_n_valid_days`, and `gait_mean_valid_hours`.

The final `gp-aggregate-cohorts` analysis step additionally masks all gait
measurements for Bologna and Ravenna rows where `gait_n_valid_days < 3`, while
preserving those three QC fields. This applies to existing sensor exports and
does not alter the standalone gait aggregation described above. Missing
valid-day counts are not treated as zero; populated measurements without counts
require a refreshed source aggregation. REDCap's clinic-measured `gait_speed`
is preserved. See [combined cohort exclusions](combined_cohorts.md#minimum-three-valid-days-or-nights).

### Sleep

The standalone sleep aggregation does not apply a new hours-per-night threshold. It reads `sleep_output_all_guiders.csv`, keeps rows with a recognized guider source, and if present keeps only rows where `spt_found` is true. The final cohort analysis requires at least 3 eligible nights per guider.

Sleep is summarized separately for each guider:

- `sleep_diary`
- `HDCZA`
- `lower_back_tib`

For each guider, the export reports `*_n_valid_nights` and mean/median sleep metrics. Final cohort assembly masks that guider's measurements below 3 nights while retaining the count and participant row.

Important upstream sleep thresholds:

- SIB sleep bouts use a minimum sustained-inactivity bout duration of `5 min`.
- HDCZA SPT detection uses 5-second angle epochs.
- HDCZA removes no-movement blocks of `<= 30 min`, bridges gaps `< 60 min`, and keeps the longest resulting block.
- The default HDCZA threshold is `10th percentile * 15` of the 5-minute rolling median absolute z-angle change, clamped to `0.13-0.50`.

If `lower_back_tib` guider windows are generated by this repo's lower-back TIB pipeline, that upstream TIB QC has additional night-wear requirements: at least `8 h` wear in the 18:00-11:00 night window, maximum continuous night nonwear `< 3 h`, at least `85%` wear inside the TIB window, TIB duration between `3 h` and `14 h`, and at least `4 h` overlap with the night window.

### Physical Activity / Activity Intensity

The activity-intensity stage builds 5-second ENMO epochs, marks missing and nonwear epochs as invalid, and summarizes each analysis day. A day is valid when:

```text
valid_hours >= 18.0
```

By default, the stage requires at least `3` valid days. If more are available, only the first `7` valid days are retained. If the retained epochs do not overlap any selected sleep-period-time window, the activity stage is skipped.

In the combined wrist runner, activity-intensity SPT windows are selected from sleep output with priority `sleep_diary`, then `HDCZA`. Lower-back TIB is summarized in sleep outputs but is not used for activity-intensity SPT segmentation.

The aggregation module reads `activity_intensity_summary.csv` and carries forward the pipeline summary fields, including:

- `activity_n_valid_days`
- `activity_day_window_min_mean`
- observed and imputed minutes
- inactive, light, moderate, vigorous, and MVPA minutes

No additional minimum-hour rule is applied at aggregation time.

### Circadian

The circadian stage builds a 1-minute ENMO table and applies the same default day-level validity as activity:

```text
valid_hours >= 18.0
min_valid_days = 3
max_days = 7
```

By default, full 24-hour days are not required. If `max_nonwear_hours` is supplied, the effective wear threshold becomes:

```text
min_wear_hours = 24 - max_nonwear_hours
```

The circadian aggregation step itself does not re-filter by hours. It appends the first row of `circadian_metrics.csv` into the sleep/circadian aggregate output.

### Cardio

There are two cardio-related paths.

BeliefPPG heart rate:

- Splits at gaps in either ACC or PPG. Each stream uses the smaller of the configured `1 s` gap limit and `1.5 / frequency` seconds.
- Trims portions to common temporal coverage and keeps portions, including the first and last, only if they are at least `10 min`.
- Runs BeliefPPG inference only on ACC/PPG portions at least `5 min` long.
- Aggregation trims HR data to the tracker visit window when available, uses selected sleep windows, filters nonpositive and participant-level outlier HR values, then computes median day/night HR and HR dip.

HRV:

- Uses selected SPT windows from the sleep pipeline, with guider priority `sleep_diary`, then `HDCZA`, then `lower_back_tib`.
- Detects acceleration bursts and builds quiet periods between bursts.
- Keeps quiet periods at least `1 min`.
- Uses whole quiet periods when they are `1-5 min`; otherwise uses `5 min` windows stepped every `1 min`.
- Requires at least `30` beats per HRV window.
- Processes only common continuous ACC/PPG intervals within each sleep window; a missing sample ends the interval at the configured sampling frequency.
- Retains interpolation and exports the affected-window flag, filled interval count and interpolated fraction. No fraction-based exclusion is applied.
- Aggregation filters nonpositive and participant-level outliers for `rmssd`, `mean_hr`, `sdnn`, and `PIP`, using median +/- `3 * IQR`, then computes participant-level medians.

Neither cardio path uses the wrist nonwear table directly, and neither imputes nonwear periods as wearable time.

## Nonwear Detection

### Wrist Empatica Nonwear

The default GP wrist nonwear method is `empatica_detach`.

It combines two sources:

1. Charging gaps: accelerometer timestamp gaps `> 60 s` are treated as nonwear.
2. DETACH nonwear: DETACH is run on non-charging chunks.

Chunk handling:

- The first non-charging chunk is kept.
- Intermediate non-charging chunks shorter than `10 min` are skipped.
- The last non-charging chunk is kept only if it is at least `10 min`.
- DETACH is only run on chunks with at least `5 min` of accelerometer samples and nonempty temperature data.
- Final charging and DETACH intervals are merged; intervals separated by `<= 10 min` are merged into one nonwear window.

DETACH defaults:

- 1-minute acceleration standard deviation windows.
- Candidate nonwear requires at least `2` axes below `8 mg`.
- At least `90%` of the next `5 min` must satisfy the axis criterion.
- Low temperature cutoff: `26 C`.
- High temperature cutoff: `30 C`.
- Temperature decrease rate for start: `< -0.2 C/min`.
- Temperature increase rate for end: `> 0.1 C/min`.

### Lower-Back Nonwear

Lower-back preprocessing uses a Van Hees / GGIR-like 2013 nonwear detector by default.

Defaults:

- 15-minute medium epochs.
- 60-minute long windows centered on each medium epoch.
- An axis counts as nonwear when both acceleration range `< 50 mg` and standard deviation `< 13 mg`.
- At least `2` axes are required.
- GGIR-style edge correction and short-wear-gap relabeling are applied.

The resulting nonwear bouts are converted into a boolean `wear` mask. Gait processing uses that mask to exclude nonwear and separate wear segments. Aggregation uses `nonwear_time_minutes` and `valid_hours`; it does not fill lower-back nonwear values.

Lower-back time-in-bed has one separate correction used only for TIB posture/QC: nonwear bouts up to `2 h` can be flipped back to wear when lying is present immediately before and after the gap within a `2 min` tolerance. This is a TIB-specific wear-mask correction, not gait imputation.

## How Nonwear Is Imputed

The pipeline does not impute raw accelerometer, temperature, or PPG files. Imputation happens only in derived analysis tables after invalid data have been flagged.

Wrist circadian:

- Invalid 1-minute ENMO values are missing minutes or minutes overlapping nonwear.
- After selecting valid days, invalid minutes are replaced by the mean ENMO at the same minute-of-day across the retained valid days.
- If no valid mean exists for that minute-of-day, the fallback value is `0.0`.
- The imputed flag is retained and daily imputed minutes are exported.

Wrist activity intensity:

- Invalid 5-second ENMO epochs are missing epochs or epochs overlapping nonwear.
- After selecting valid days, invalid epochs are replaced by the mean ENMO at the same epoch-of-day across retained valid days.
- If no valid mean exists for that epoch-of-day, the fallback value is `0.0`.
- Observed and imputed minutes are reported separately in the daily and summary outputs.

Sleep HDCZA:

- Missing or nonwear 5-second z-angle epochs are invalid.
- Invalid angle epochs are imputed from the same epoch-of-day across days.
- If some epoch-of-day means are missing, the mean profile is linearly interpolated across clock time.
- With the default `haspt_ignore_invalid = False`, HDCZA uses the imputed angle values during invalid epochs.

Sleep SIB:

- Nonwear angle epochs are set to null.
- There is no SIB imputation; nulls break sustained-inactivity bouts.
- Sleep summaries also report nonwear duration inside each guider window as `nonwear_dur_in_guider`.

Gait and lower-back:

- Gait nonwear is not imputed.
- Nonwear contributes to `nonwear_time_minutes`, which reduces valid hours for gait amount-feature aggregation.
- The TIB-specific short-nonwear correction described above can relabel short nonwear gaps as wear for posture/TIB processing only.

Cardio:

- Heart-rate and HRV pipelines do not use the wrist nonwear windows directly.
- BeliefPPG skips portions that are too short after gaps.
- HRV removes acceleration-burst periods and uses quiet periods.
- HRV artifact handling interpolates cleaned IBI values after beat/artifact filtering; this is beat-signal cleanup, not nonwear imputation.

## Main Source Files

- `src/gp_pipeline/aggregation/gait.py`
- `src/gp_pipeline/aggregation/sleep.py`
- `src/gp_pipeline/aggregation/activity_intensity.py`
- `src/gp_pipeline/aggregation/heart_rate.py`
- `src/gp_pipeline/aggregation/hrv.py`
- `src/gp_pipeline/wrist/circadian/circadian_pipeline_gp.py`
- `src/gp_pipeline/wrist/circadian/circadian_pipeline.py`
- `src/gp_pipeline/wrist/circadian/activity_intensity.py`
- `src/gp_pipeline/wrist/sleep/sleep_pipeline_gp.py`
- `src/gp_pipeline/wrist/sleep/vh2015_sib.py`
- `src/gp_pipeline/wrist/sleep/vh2018_spt.py`
- `src/gp_pipeline/wrist/nonwear/nimbaldetach.py`
- `src/gp_pipeline/lower_back/preprocessing.py`
- `src/gp_pipeline/lower_back/nonwear/vanhees2013.py`
- `src/gp_pipeline/lower_back/pipeline.py`
- `src/gp_pipeline/wrist/heart_rate/pipeline.py`
- `src/gp_pipeline/wrist/heart_rate_variability/pipeline.py`
