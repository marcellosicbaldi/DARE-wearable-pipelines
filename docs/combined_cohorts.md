# Combined DARE-FALLSPREDICT GP and DARE-FALLSPREDICT dataset

`gp-aggregate-cohorts` is the final local assembly step. It reads the completed
sensor exports from each site, processes each site's REDCap with its own
FRAT-up files, applies the specified visual exclusions, minimum-data masks and column removals, and
combines the rows using the union of the retained columns. It does
not access the DARE-FALLSPREDICT server or recompute sensor features.

## Run

Activate the pipeline Python environment, then run from the project root:

**After the three-day/night rule update, first refresh DARE-FALLSPREDICT GP's sensor
aggregation.** Older exports do not contain the separate HR counts or
metric-specific HRV counts needed to enforce the rule. This reads existing
Silver outputs; it does not rerun raw sensor processing. DARE-FALLSPREDICT's existing
gait/sleep/activity exports already include the required counts.

```bash
cd "/path/to/DARE-wearable-pipelines"
PYTHONPATH=src python -m fallspredict_gp_pipeline.aggregation.overall \
  --silver-root "~/dare-data/silver" \
  --visit T0 --sleep-method both \
  --output-dir "~/dare-data/silver/aggregation"
```

Then rebuild the final dataset:

```bash
cd "/path/to/DARE-wearable-pipelines"

PYTHONPATH=src python -m fallspredict_gp_pipeline.aggregation.cohorts \
  --sleep-method mean \
  --exclusions-config configs/analysis_exclusions.local.toml \
  --output-dir "~/dare-data/gold/aggregation"
```

After reinstalling the editable package (`pip install -e .`), the equivalent
entry point is `gp-aggregate-cohorts`. The module command works without
reinstalling the command-line entry points. Use `--sleep-method median` to build
the corresponding alternative dataset; mean and median exports are never
stacked together as duplicate participant rows.

The main result is `gold/aggregation/fallspredict_combined_T0.csv`. The same folder
also receives `fallspredict_bologna_T0.csv` and `fallspredict_ravenna_T0.csv` with
**exactly the same columns, in the same order**, as the combined file. For a
median run the filenames retain `_sleep_median`, e.g. `fallspredict_combined_T0_sleep_median.csv`.
All files saved by this assembly step, including audits, start with `fallspredict_`.
Mean is the default and has no sleep suffix. Existing files without the prefix
or with the old `_sleep_mean` suffix are not rewritten or deleted by the new run.

To omit participant IDs from the saved CSVs, run:

```bash
PYTHONPATH=src python -m fallspredict_gp_pipeline.aggregation.cohorts --no-manual-exclusions --save-subject-id FALSE
```

`--save-subject-id TRUE` is the default; values are case-insensitive. IDs are
used internally for matching, visual exclusions and aggregation, then removed
at save time from **all** CSVs produced by this command, including audits.
The removed column names are `subject`, `subject_id`, `patient_id`, `record_id`,
`id` and `source_subject` (case-insensitive). Their rows are also omitted from
column coverage. No replacement ID or CSV index is written; `group`, `visit`,
row order and all other values are retained.

Output filenames stay the same, so this replaces the corresponding exports in
the selected directory. Use `--output-dir <folder>` to keep a separate version.
Source files and older exports under different names are untouched. This is
column omission, not full anonymisation: dates, free text, source paths and
source-row references remain. Audit rows lose their participant join keys.
For Python calls, use `write_combined_exports(save_subject_id=False, ...)`.

Legacy DARE-FALLSPREDICT GP files with no nonwear column are interpreted as wear-filtered;
newer files report recorded hours and nonwear separately.

All defaults are relative to
`~/dare-data` (expanded to the current user's home directory):

| Data | Default location | Override |
| --- | --- | --- |
| DARE-FALLSPREDICT GP sensors | `silver/aggregation` | `--bologna-sensor-dir` |
| DARE-FALLSPREDICT sensors | `silver/aggregation/Ravenna` | `--ravenna-sensor-dir` |
| DARE-FALLSPREDICT GP REDCap | `REDCap/Bologna/fallspredict_data.csv` | `--bologna-redcap-csv` |
| DARE-FALLSPREDICT REDCap | `REDCap/Ravenna/fallspredict_data.csv` | `--ravenna-redcap-csv` |
| Both cohorts' FRAT-up files | `gold/fratup` | `--fratup-dir` |
| Combined output | `gold/aggregation` | `--output-dir` |

All inputs are read-only. New exports replace earlier files with the same output
names on subsequent runs; choose a different output folder to retain snapshots.
An output cannot overwrite a selected input file. `--no-fratup` explicitly
disables external scores/inputs for both cohorts.

## Selecting sensor files

Each cohort first looks for `overall_T0_sleep_mean.csv` (or `.xlsx`; replace
`mean` with `median` for the alternative). An overall export takes precedence
over individual domain files, and the selected paths are saved in the source
manifest. Keep that file current if it is present.

Without an overall export, these three domain files are required:

- `sleep_T0_mean.csv` or `.xlsx` (or the requested median version).
- `gait_T0_mean.csv` or `.xlsx`.
- `activity_intensity_T0.csv` or `.xlsx`.

This directly supports the current DARE-FALLSPREDICT folder's four Excel files: the
selected sleep variant plus gait and activity. A workbook must contain one
sheet, as the supplied workbooks do. Having both `.csv` and `.xlsx` for a
selected filename is an ambiguity error. Other files are not guessed or
silently merged. Loading retains the selected exports' columns and rejects
overlapping non-key columns between domain tables. Final analysis column
removals and measurement exclusions are applied after the clinical/sensor join.

## Participant identity and missing values

Every final row is internally keyed by **`group`, `subject`, `visit`**;
`subject` is omitted from exports only when `--save-subject-id FALSE` is used:

- `group` is explicitly `BO` or `RA`, assigned from the selected cohort.
- `subject` is resolved from REDCap `patient_id`; event rows are linked within
  their `record_id`. Record IDs, including DARE-FALLSPREDICT's `FP...` identifiers, are
  not used as patient IDs. Numeric subject IDs normalize leading zeros and
  trailing `.0`, with a minimum display width of four digits, as in the sensor
  merge. Original identifiers remain in the event and FRAT-up input exports
  unless `--save-subject-id FALSE` is used.
- `visit` is `T0`. Derived monthly follow-up measures remain
  separate columns describing follow-up; they are not substituted for baseline
  measurements. T1 clinical aggregation is not implemented.

The two sites are processed independently before concatenation. Identical
numeric subject IDs in different cohorts cannot cross-match. Within a cohort,
joins are outer joins with unique-key validation, retaining participants who
have only REDCap or only sensors. `has_redcap` and `has_sensors` indicate whether
a source record exists, not whether all of its measurements are complete.
REDCap-only records without baseline remain in the population while their
mapped T0 clinical values stay missing. Their raw events are exported separately.

Unavailable fields are missing values, never filled with zero. CSV exports use
the literal `NaN` marker, which `pandas.read_csv` reads as missing by default.
For example, the current DARE-FALLSPREDICT exports have no HR/HRV columns, so those
DARE-FALLSPREDICT GP columns are retained in the combined schema and are NaN on DARE-FALLSPREDICT
rows. The presence flags are the exception: a missing source is explicitly `0`.
The column coverage file distinguishes a column absent from one site's schema
from a present column whose observations happen to be missing.

## Measurement exclusions and final column selection

Participant IDs are never embedded in the published package. Supply a private
`--exclusions-config /path/to/analysis_exclusions.local.toml`, or explicitly use
`--no-manual-exclusions`. The two options are mutually exclusive and one is
required. Python callers use `exclusions_config=...` or
`no_manual_exclusions=True`. Minimum-data masks and column removals apply in
either case. The configuration format is shown in
[`analysis_exclusions.example.toml`](../configs/analysis_exclusions.example.toml).
Local configuration files are ignored by Git and excluded from distributions.
The private config path is included in the local source-file audit.

These rules live in `src/fallspredict_gp_pipeline/aggregation/analysis_exclusions.py` and are
applied by the final cohort assembly command, before schema alignment and
coverage counts. They affect the combined, DARE-FALLSPREDICT GP and DARE-FALLSPREDICT analysis files
for both mean and median sleep runs. They do not modify source sensor exports,
REDCap records, FRAT-up files, or the earlier automatic per-window HRV filtering.

### Minimum three valid days or nights

`src/fallspredict_gp_pipeline/aggregation/minimum_data.py` applies the requested minimum of
**3** observations independently to each domain in both cohorts. A count of
0, 1 or 2 masks the corresponding measurements to NaN; exactly 3 passes.
It replaces the previous gait-only `== 0` mask.

| Measurements | Required coverage count |
| --- | --- |
| Wearable gait amount and quality | `gait_n_valid_days` |
| `sleep_diary_*` sleep measurements | `sleep_diary_n_valid_nights` |
| `hdcza_*` sleep measurements | `hdcza_n_valid_nights` |
| `lower_back_*` sleep measurements | `lower_back_n_valid_nights` |
| `median_hr_day` | `hr_n_valid_days` |
| `median_hr_night` | `hr_n_valid_nights` |
| `hr_dip_pct` | `hr_dip_n_valid_pairs` |
| HRV measurements and window summaries | `hrv_n_valid_nights` |
| Each of `hrv_rmssd`, `hrv_sdnn`, `hrv_mean_hr`, `hrv_PIP` | Additionally, its own `hrv_<metric>_n_valid_nights` |
| Activity intensity | `activity_n_valid_days` |

Circadian processing already requires at least 3 valid days upstream. Its
`circadian_cosinor_days` is a duration (`minutes / 1440`), not a valid-day count,
and is not used as a substitute for one. The sleep guider counts do not gate
circadian measures or a different guider's sleep measures.

The HR aggregation now counts nonmissing estimates after its existing filtering,
once per `night_id`. The daytime count refers to available daytime periods
between successive sleep windows; the first night has no preceding daytime
estimate. The dip count requires an available paired day/night estimate.
No new daily wear-hour or within-night duration threshold is introduced for HR.

The HRV aggregation now counts distinct nights with at least one retained
physiological metric, excluding nights whose measurements were all filtered
out. It also counts contributing nights separately for each metric. Multiple
windows on the same night count once. A participant with three SDNN nights
but only two RMSSD nights retains SDNN and has RMSSD set to NaN.
HR and HRV standalone exports also apply their respective minimums.

The final assembly retains all QC counts, including `gait_n_days`,
`gait_n_valid_days`, `gait_mean_valid_hours`, each sleep guider's night count,
HR counts, HRV counts and `hrv_n_windows`. Participant rows, source-presence
flags and clinical variables are preserved. In particular, REDCap's
clinic-measured **`gait_speed` is not a wearable feature** and is not masked.
The existing manual visual exclusions still apply separately.

This is a participant-level coverage rule; it does not change gait's
`valid_hours > 16` definition, recompute its quality averages, or remove
individual walking bouts. There are no new exclusion flag columns: coverage
counts record the reason for these masks.

Missing counts are not converted to zero. If measurements are populated but
their required count is missing or unreadable, assembly stops with an explicit
request to refresh that source aggregation. Already-missing measurements,
including clinical-only rows, do not require a count. This prevents older HR
exports from silently bypassing the new rule.

### RMSSD and SDNN

For DARE-FALLSPREDICT GP T0 participants listed in `manual_exclusions.rmssd_sdnn_ids`,
both `hrv_rmssd` and `hrv_sdnn` are set to NaN. Other HRV and heart-rate
metrics remain available. Numeric IDs are normalized before matching.

- `rmssd_sdnn_exclusion`: `1` for the listed DARE-FALLSPREDICT GP T0 participants, otherwise `0`.
- `rmssd_sdnn_exclusion_method`: `visual inspection` when excluded, otherwise NaN.

### Sleep, circadian and activity intensity

For DARE-FALLSPREDICT T0 participants listed in `manual_exclusions.sleep_circadian_ids`,
all sensor columns starting with these prefixes are set to NaN:

| Prefix | Data excluded |
| --- | --- |
| `sleep_diary_` | Sleep-diary-guided sensor sleep metrics and quality counts. |
| `hdcza_` | HDCZA sleep metrics and quality counts. |
| `lower_back_` | Lower-back-guided sleep metrics and quality counts. |
| `circadian_` | All retained circadian variables, including clock strings and counts. |
| `activity_` | All activity-intensity variables and valid-day counts. |

- `sleep_circadian_exclusion`: `1` for the listed DARE-FALLSPREDICT T0 participants, otherwise `0`.
- `sleep_circadian_exclusion_method`: `visual inspection` when excluded, otherwise NaN.

The sleep exclusion includes activity intensity despite the shorter flag name.
This sleep rule retains gait features (`gait_`), HRV, clinical scores such as
PSQI, FRAT-up, and follow-up variables; the separate minimum-data rules
still applies. Participant rows and `has_sensors` are retained: that
flag describes source availability, not whether a domain was excluded.

Keep study-specific identity checks in your private analysis records. Rules use cohort and T0 as well
as normalized subject ID, so an identical ID in the other cohort or a later
visit is not excluded. Flags record the specified decision even when a value
was already missing. Missing measurements alone do not set an exclusion flag.

### Columns omitted from all final analysis datasets

The following 17 columns are removed globally, including from the final column
coverage report. Other similarly named variables are retained:

```text
hdcza_spt_start_clock_h
hdcza_spt_end_clock_h
hdcza_guider_start_clock_h
hdcza_guider_end_clock_h
gait_wb_all__n_raw_initial_contacts__sum
gait_wb_all__n_turns__sum
hrv_window_length_s
hrv_source_night_id
hrv_priority_rank
hrv_PIP_percent_discarded_median
hrv_mean_hr_percent_discarded_median
hrv_rmssd_percent_discarded_median
hrv_sdnn_percent_discarded_median
circadian_MESOR_log1p_mg
circadian_Amplitude_log1p_mg
circadian_Acrotime_hour
circadian_Cosinor_n_days_used
```

`fallspredict_cohort_summary_T0.csv` (or `fallspredict_cohort_summary_T0_sleep_median.csv`) reports `rmssd_sdnn_exclusions` and
`sleep_circadian_exclusions` per cohort. Column coverage counts reflect the
values remaining after exclusions.

```python
import pandas as pd

dataset = pd.read_csv(
    "~/dare-data/gold/aggregation/fallspredict_combined_T0.csv",
    dtype={"group": "string", "subject": "string", "visit": "string"},
)
assert not dataset.duplicated(["group", "subject", "visit"]).any()
```

## Clinical mapping and raw event fields

Both cohorts use the existing mapped T0 variables, questionnaire calculations,
disease/ATC definitions, fall verification, and monthly follow-up rules described
in [REDCap processing](redcap_processing.md). This applies the supplied notebook
specification to the shared source field names; it is not a separate validation
of DARE-FALLSPREDICT's clinical codebook. Historical mapping limitations remain visible
in the mapping and codebook-issue exports. If a site's code definitions differ,
use `--bologna-codebook` or `--ravenna-codebook` for that site's replacement JSON.

FRAT-up selects `fratup_input_BO_T0.csv` and `result_BO_T0.csv` for DARE-FALLSPREDICT GP,
and `fratup_input_RA_T0.csv` and `result_RA_T0.csv` for DARE-FALLSPREDICT. The alternative
`results_` spelling is accepted. Both input and score files require identifiers;
IDs are checked only against the selected site's REDCap. Private R code remains
external. Distinct FRAT-up input fields are included in the same column union.

The combined and cohort-specific analysis CSVs contain mapped clinical variables,
FRAT-up scores/inputs, derived monthly follow-up measures, sensor features, and
source-presence flags. Raw fields expanded into columns named
`redcap__<event>__<field>` are excluded as requested. This removes the block of
1,742 columns previously produced from DARE-FALLSPREDICT's 134 extra fifth-fall fields.

All original fields remain in the separate long-format `fallspredict_redcap_events.csv`,
except the ID columns when `--save-subject-id FALSE` is used.
Fifth falls still contribute to the existing fall counts, date allocation, and
`verified` measures. The union of mapped clinical, FRAT-up and sensor columns is
still retained with NaN for the cohort without a field; columns are not truncated
by their position relative to `has_sensors`.

The obsolete `extra_redcap_fields.csv` mapping is no longer generated. A copy
from a previous run is not rewritten or deleted and is not part of the new
exports. Derived follow-up outcomes remain in the analysis dataset, so select
baseline predictor columns explicitly for prediction models.

## Audit exports

| Export | Contents |
| --- | --- |
| `fallspredict_cohort_summary_T0.csv` | Cohort counts, matches, unmatched participants, FRAT-up coverage, and visual-exclusion counts. |
| `fallspredict_column_coverage_T0.csv` | Column presence and nonmissing counts for each cohort. |
| `fallspredict_source_files_T0.csv` | Exact local input paths selected for the run. |
| `fallspredict_redcap_events.csv` | Union of all original REDCap rows/fields with cohort identity. |
| `fallspredict_redcap_falls.csv` | Individual fall records and verification, distinguished by cohort. |
| `fallspredict_redcap_fall_monthly.csv`, `fallspredict_redcap_followup_monthly.csv` | Calendar and source-event follow-up tables. |
| `fallspredict_redcap_mapping.csv`, `fallspredict_redcap_codebook_issues.csv` | Clinical mapping assumptions and dictionary issues by cohort. |
| `fallspredict_redcap_fratup_inputs.csv`, `fallspredict_redcap_fratup_import.csv` | Preserved external inputs and import decisions by cohort. |

The summary, coverage and source filenames append `_sleep_median` for a median run. The
other audit exports concern the same clinical data regardless of sleep variant.

## Validation and execution status

Read-only inspection of the supplied files found 200 DARE-FALLSPREDICT GP and 538 DARE-FALLSPREDICT
baseline participants. DARE-FALLSPREDICT GP's previous sensor export contains 197 matched
participants. DARE-FALLSPREDICT's local domain exports cover 529 distinct participants,
all matching its REDCap patient IDs. With these inputs the expected combined
population is 738 participants, including 12 without sensor rows.

The new assembly command has **not been run on participant data**, following the
request to provide execution instructions. Existing exports were not changed.
The source schema and identifier checks above are inspections, not claims that
the new combined dataset has already been generated.

The synthetic regression tests cover cohort-specific FRAT-up selection,
overlapping IDs across sites, unknown/missing/duplicate identifiers, NaN padding,
identical output schemas, exclusion of raw event columns from analysis exports,
source-only participants, alternative sleep variants, domain-file discovery,
the existing clinical/fall mappings, synthetic exclusion IDs, padded IDs,
cohort/visit isolation, method labels, the exact column-removal list, and masking
in both mean and median analysis exports. Coverage tests check the 2/3 boundary,
independent guider and cardio counts, filtered-out HRV nights, missing-count
validation, preservation of clinical gait speed, and identical cohort masks.
Run them with:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
