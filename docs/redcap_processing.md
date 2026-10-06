# Bologna and Ravenna REDCap processing

The extractor preserves a study-specific 185-column T0 schema. Clinical
scoring conventions are documented below; participant values are computed
from REDCap or imported from private FRAT-up CSVs. An optional private
reference file can be compared but never supplies replacement values.
Additional FRAT-up inputs follow the original columns.

There are now mappings for 183 original columns, including the externally
computed FRAT-up score. Its private R algorithm remains outside this project.
`fallen_6m` / `followup_6m` remain deferred under the earlier instruction to
preserve all follow-up rather than select an exact six-month endpoint.
Implemented does not mean every participant has a value: missing source inputs
remain visible. Historical scoring and codebook issues are described below.

The same mapped T0 schema now supports `--cohort BO` (default) and `--cohort RA`.
Each cohort selects its own default REDCap path and FRAT-up pair. To assemble
both sites' local sensor exports into a single dataset, use the separate final
[combined cohort assembly](combined_cohorts.md) command. It aligns the mapped
clinical, FRAT-up, follow-up and sensor columns with missing values where a field
is absent. Raw REDCap event fields remain in a separate event export and are not
expanded into additional analysis columns.

## Run and merge with sensors

After installing the project in your pipeline environment (`pip install -e .`):

```bash
gp-aggregate-redcap \
  --redcap-csv "~/dare-data/REDCap/Bologna/fallspredict_data.csv" \
  --reference-csv "/path/to/private/reference.csv" \
  --output-dir outputs/redcap
```

Alternatively use `PYTHONPATH=src python -m gp_pipeline.aggregation.redcap`
with the same arguments from the project root. Only pandas and NumPy are
required. The reference argument is optional and performs comparison only.
The default source is now `REDCap/Bologna/fallspredict_data.csv`, relative to the generic data root. With `--cohort RA`, the default changes to
`REDCap/Ravenna/fallspredict_data.csv`. An explicitly supplied path is used as given;
there is no automatic source substitution.
To merge an existing T0 sensor export, add:

```bash
--sensor-csv "~/dare-data/silver/aggregation/overall_T0_sleep_mean.csv"
```

To recompute sensor aggregations and join clinical and monthly fall data:

```bash
gp-aggregate-all \
  --silver-root "~/dare-data/silver" \
  --visit T0 \
  --redcap-csv "~/dare-data/REDCap/Bologna/fallspredict_data.csv" \
  --output-dir outputs/combined
```

Omitting `--redcap-csv` preserves the sensor-only workflow. T0 is the only
implemented clinical schema; T1 and all other raw events remain preserved.
Joins normalize IDs, validate unique `subject, visit` keys, reject overlapping
non-key columns, and retain participants without sensors.

```python
from gp_pipeline.aggregation.redcap import (
    process_redcap, aggregate_redcap, merge_redcap_with_sensors,
)

result = process_redcap(redcap_csv)
clinical = result.clinical       # Original 185 columns, then extra FRAT-up inputs
falls = result.falls             # Individual reported falls; includes verified
calendar = result.fall_monthly   # All 12 months for every participant
calls = result.followup_monthly  # Existing monthly events and their responses
events = result.events          # Every original field from every event
fratup_inputs = result.fratup_inputs  # All supplied inputs, with resolved IDs
fratup_audit = result.fratup_import   # Linkage method and reused/added columns

clinical_for_merge = aggregate_redcap(redcap_csv)
combined = merge_redcap_with_sensors(clinical_for_merge, sensor_dataframe)

# Ravenna alone (use the combined assembly command for both sites):
ravenna = process_redcap(cohort="RA")
```

## Local FRAT-up results and inputs

Both aggregation commands automatically read
`~/dare-data/gold/fratup` when processing REDCap.
Use `--fratup-dir <folder>` to change it, or `--no-fratup` to explicitly leave
the score missing. The Python equivalents are `fratup_dir=folder` and
`fratup_dir=None` on `process_redcap`, `aggregate_redcap`,
`write_redcap_exports`, `aggregate_all`, and `write_all_exports`. Sensor-only
runs do not access this folder. No R code is copied, distributed, sourced, or
executed by the pipeline.

For Bologna T0, the required pair is:

- `fratup_input_BO_T0.csv`: participant identifiers and the risk-factor inputs
  supplied to FRAT-up.
- `result_BO_T0.csv`: participant identifiers and `fratup`. The spelling
  `results_BO_T0.csv` is also supported; having both is an ambiguity error.

With `--cohort RA` (Python `cohort="RA"`), the corresponding `_RA_T0` filenames
are selected. Files for other cohorts or visits are not mixed into the selected
cohort's T0 data. Scores are imported
unchanged as probabilities in `[0, 1]`, not multiplied by 100 or recomputed.
Blank/`NA` scores and baseline participants without a result remain missing;
coverage appears in `redcap_fratup_import.csv` and incomplete coverage raises a
warning. Missing files, invalid values, duplicate/unknown IDs, and conflicting
identifiers raise errors rather than silently returning an empty score column.

**Identity-based matching.** Both input and result files must identify participants through
`patient_id`/`subject`, or `record_id` (`id` is accepted as a record-ID alias).
Record IDs are resolved through the REDCap baseline identity map and are never
treated as patient IDs. When multiple identifiers are present they must agree.
Input and result files can be reordered independently; their participant sets
must match. Their identifiers are resolved separately through REDCap. The
updated Bologna and Ravenna T0 files contain both `record_id` and `patient_id`; the loader
checks that these agree. Files without identifiers are rejected. There is no
row-order fallback, and demographics are not used to establish identity.
Identifiers are preserved in the dedicated input export and excluded from the
risk-factor columns added to the clinical table.

**Avoiding duplicate columns.** Six equivalent inputs can reuse existing
clinical columns after checking every imported row, including missingness:

| FRAT-up input | Clinical column | Encoding |
| --- | --- | --- |
| `sex` | `sex` | `0/1` maps to `M/F`. |
| `age` | `age` | Unchanged. |
| `livingalone` | `living_alone` | Unchanged. |
| `hfallYN` | `history_falls` | Unchanged. |
| `dizziness` | `dizziness` | Unchanged. |
| `numberofmed` | `daily_medications` | Unchanged. |

Other inputs are appended as `fratup_input_<original_name>`. Binary input flags
remain distinct from clinical questionnaire totals or disease/ATC categories:
for example, `cognitionimpairment` is not interchangeable with `mmse`.
A differing shared input is also preserved under this prefix
and its discrepancy count is audited; it does not overwrite or backfill REDCap.

All 26 original risk-factor inputs and supplied identifiers, including the six reused fields, are retained in
`redcap_fratup_inputs.csv`, together with normalized `subject` and `source_row`
(one-based data row, excluding the header). If the source already has a
`subject` column it is retained as `source_subject`. The import audit records
source paths, linkage method, reuse/addition decisions, and score coverage.
The original 185-column order is unchanged; the last generated exports have 20 additional
input columns (205 clinical columns total). Reference comparison still checks
only the original 185-column schema. Inputs also flow into the combined sensor
exports. Generated participant data stays in ignored `outputs/` files.

## Review of the colleague's fall code

The central approach makes sense for retrospective reporting: assign a fall
to a month using its **occurrence date**, while retaining the REDCap event in
which staff recorded it. Excluding baseline fall history is also appropriate.
The relative-calendar-month calculation is reasonable; the implementation here
expresses its month-end rule explicitly with anniversary boundaries.

The supplied code should not be used unchanged:

| Issue | Consequence | Implemented change |
| --- | --- | --- |
| Global `ffill()` of IDs and baseline dates | Row order or a missing baseline date can attach another participant's information. | Resolve IDs within `record_id`; select each subject's actual baseline date. |
| `drop_duplicates()` takes the first row, not necessarily baseline | Follow-up may be used as the starting record. | Select `baseline_arm_1` explicitly; reject duplicate baselines. |
| Broad date/completion lists paired with `zip()` | Column order can pair a fall with the sensor questionnaire or another fall's status; unequal lists are truncated. | Five explicit date/form pairs, listed below. |
| `col_comp` is never used | Dated entries are counted without verification. | Each reported fall has nullable `verified`, based on its own form. |
| `f'mese_{m}' in event_name` | `mese_1` matches `mese_10`, `mese_11`, and `mese_12`. | Anchored full event-name matching. |
| Placeholder `chiamata_complete` or `followup_complete` | The saved export has no `chiamata_complete`, and `followup_complete` is entirely blank. `follow_up_complete` is the six-month laboratory assessment. | Use `domande_caduta_complete` together with a valid `cadute_mese` response for routine assessment completion. |
| All monthly fall counts/binaries start at 0 | Missing calls/dates become apparent no-fall months. | Separate recorded-entry counts from nullable outcomes and call coverage. |
| Only dated falls are retained | Reported falls with missing/invalid dates disappear. | Retain them with an allocation reason; never invent a date/month. |
| Repeated reporting is not examined | A re-entered fall may be counted twice. | Flag same-subject/same-date candidates; do not remove potentially distinct same-day falls automatically. |

A completed form does **not** establish that the call occurred on time.
The export has no usable contact timestamp establishing retrospective delay.
`recorded_in_different_month` is an event/occurrence mismatch, not proof of a
late call. Completing a later fall interview verifies that reported fall under
the operational rule; it does not certify every intervening month's coverage.

## Fall verification and explicit form pairing

| Slot | Date field | Completion field |
| --- | --- | --- |
| 1 | `date_fall` | `domande_caduta_complete` |
| 2 | `date_fall_c2` | `domande_seconda_caduta_7955_complete` |
| 3 | `date_fall_c3` | `domande_terza_caduta_complete` |
| 4 | `date_fall_c4` | `domande_quarta_caduta_complete` |
| 5 | `date_fall_c5` | `domande_quinta_caduta_complete` |

The new **`verified`** column is in **`redcap_falls.csv`**, at individual-fall
level: 1 for status 2 (Complete), 0 for status 0/1 (Incomplete/Unverified), and
missing if the field/status is absent or invalid. This is the requested
operational definition, not independent adjudication. Raw `form_status`, field
name/presence, source row, and event remain available for audit. Verification
and calendar allocation are independent: a complete form may lack a valid date.

A slot becomes a fall record when it has a nonblank date, is included in
`falls_encounter`, is the first slot with `cadute_mese=1`, or is an additional
slot whose form is started/completed (status 1/2). First-form completion alone
never creates a fall: that form is also completed for no-fall calls. Missing
dates remain visible. Counts beyond five slots are flagged in
`n_reported_falls_beyond_slots`; the original count is preserved.

Up to five date/form pairs are supported; an absent column is not an incomplete form.

## Occurrence months, calls, and missingness

Month `m` is the interval:

```text
[baseline + (m-1) calendar months, baseline + m calendar months)
```

Boundaries are anchored to the original baseline date. January 31 gives
February 28/29 and March 31; these are not 30-day bins. The baseline date
belongs to month 1; the 12-month anniversary is excluded. ISO dates and explicit
Italian `DD/MM/YYYY` dates are accepted. Invalid, pre-baseline, or out-of-window
dates are retained and labelled but not allocated.

`fall_month` is the occurrence month; `source_event_month` comes from the exact
REDCap event name, including `mese_6__follow_up_arm_1`. A month-2 fall entered
at month 5 contributes to month 2's date-based counts and retains its month-5
source provenance. Baseline-history falls never enter prospective counts.

Event-based `fall_occurred` / `n_falls_reported` retain `cadute_mese` /
`falls_encounter` at the recording event. They are not silently replaced or
reassigned. `_c2`–`_c5` mean additional falls, not follow-up months.

`routine_call_verified` is 1 when the first questionnaire has status 2 and a
valid binary fall response. Known incomplete forms, or complete forms without
a binary response, give 0. Missing status/event gives missing. This is a proxy
for a completed assessment stored at an event, not a timely-call indicator.

`redcap_fall_monthly.csv` includes every participant × 12 months:

| Column | Meaning |
| --- | --- |
| `routine_event_present`, `routine_form_status`, `routine_call_verified` | Recording-event presence and assessment completion. |
| `fall_occurred`, `n_falls_reported` | Original event response/count, including blanks. |
| `n_dated_falls` | Dated, in-window fall records allocated to the month. Zero means no such records, not necessarily no falls. |
| `n_verified_falls`, `n_unverified_falls`, `n_unknown_verification_falls` | Those dated records split by verification. |
| `n_falls_observed`, `fall_observed` | Positive if dated falls exist; zero only with a completed explicit no-fall response and no flag/count conflict; otherwise missing. These may undercount undated or unreported falls. |
| `date_event_conflict` | Dated falls occur in a month whose original event says no falls. The dated falls remain included. |
| `n_possible_duplicate_entries` | Same-subject/same-date records needing review; counts retain them pending adjudication. |

The wide export and sensor merge carry these measures with `_m01`–`_m12`
suffixes. `n_fall_records_unallocated` counts retained falls without an
in-window month. `n_fall_records_verified_total` counts complete fall records
regardless of date allocation. These are not fixed-period endpoints.
`n_followup_months_observed` retains its earlier meaning: valid binary responses
regardless of form completion. Blank raw fall counts remain blank.

## Clinical mappings from the corrected notebook

- Cell 6 supplies demographics, 26-item MMSE, seven-item short FES-I, CES-D,
  six ADL and eight IADL items. Sums require all inputs. Unlike the notebook's
  final blanket `fillna(0)`, missing age, scores, and dropout flags remain
  missing. Injurious history is zero when previous falls explicitly equals
  zero; otherwise it follows the injury response.
- The commented cell-6 equation is implemented as `mmse_corrected`:
  `2.4 * log10(93.9 - age) * education_years**0.29 + 22.1`, with valid
  nonnegative inputs and age < 93.9. It matches the May reference exactly.
  **It does not use measured MMSE.** The historical column name is retained;
  this does not validate an adjustment to an individual's measured score.
- Cell 33 supplies all 60 disease categories and cell 37 all 94 ATC indicators,
  bundled in `src/gp_pipeline/redcap/codebook.py`. All 15 baseline disease
  slots and 20 medication slots are inspected, including slots omitted by the
  notebook. Totals sum category flags, not raw slots.
- Default category coding reproduces the notebook: any matched code gives 1,
  otherwise 0. Zero means no matching recorded code, not confirmed clinical
  absence. Entirely absent column families remain missing. Observed unmapped
  codes are exported for review.
- WFG follows cell 29. PSQI follows the final definition, cell 44, which
  overrides the earlier component-based definition in cell 41. These are historical compatibility conventions.

Historical scoring quirks remain explicit: PSQI converts wake `.30` to `:50`,
uses `sleep_problems_other_ynf_b`, and gives some missing components default/zero
points. WFG's `SI`/`NO` checks do not recognize lowercase `sì`/`no`, uses
CFS >= 4 and the faster walk for 0.8 m/s, while exported gait speed uses the
slower walk. Missing prior-fall history gives `ND`. These are compatibility
rules, not new clinical instrument validations.

### Codebook issues requiring review

The notebook gives **identical codes 957–975 inclusive to both
`t0_Solid_neoplasms` and `t0_Venous_Lymp_dis`**. Their ICD comments describe
different conditions, so this appears to be a copy/paste error. The correct
venous/lymphatic range cannot be established from the notebook alone. Both
historical indicators are retained and the overlap is exported explicitly.
`n_morbidities` counts both when that range is present.

Unmapped source codes and affected-subject counts are written to the private
`redcap_codebook_issues.csv` audit. Resolve unknown codes before treating
affected categories or totals as validated measures.

A replacement JSON can be supplied with `--codebook`, or `--redcap-codebook`
with `gp-aggregate-all`. It must identify its `source`. Each supplied
`diseases` or `atc` object must map every target category to a list of positive
integer REDCap codes. Wholly empty families are rejected. With replacement
codebooks, unrecognized source codes leave otherwise-negative flags/totals
missing; known positives remain usable. Omitted families remain unresolved.

The notebook redefines `info_cadute` several times with different date-based,
event-based, and six-consecutive-month rules. None is silently selected as
the definitive six-month endpoint; all monthly observations remain available.

## Export inventory and source preservation

| File | Contents |
| --- | --- |
| `redcap_clinical_T0.csv` | Original 185-column clinical prefix, with imported `fratup`, followed by additional inputs. |
| `redcap_fratup_inputs.csv` | All local input fields, resolved subject, and original data-row number. |
| `redcap_fratup_import.csv` | Source files, linkage method, coverage, and reused/added input columns. |
| `redcap_events.csv` | Every original row/field, plus resolved subject. |
| `redcap_falls.csv` | Individual falls, `verified`, dates, allocation/QC, provenance. |
| `redcap_fall_monthly.csv` | Complete participant × 12-month occurrence/ascertainment table. |
| `redcap_followup_monthly.csv` | Existing monthly events, responses, completion, conflicts, slot overflow. |
| `redcap_followup_wide.csv` | Subject-level event and occurrence measures for all 12 months. |
| `redcap_mapping.csv` | Source fields, rules, status, and availability per clinical column. |
| `redcap_codebook_issues.csv` | Identical category mappings and observed unmapped codes. |
| `redcap_reference_comparison.csv` | Optional differences from the May reference. |
| `redcap_clinical_sensors_T0.csv` | Optional clinical/follow-up/sensor outer join. |

Load IDs as strings: `pd.read_csv(path, dtype={"subject": "string"})`.
IDs are resolved through `record_id` independently of row order; global
forward-fill and substituting record_id for patient_id are not used. Ambiguous
identities and duplicate baselines/months raise errors. A previous event export
is accepted as an explicit snapshot only after its subject column is revalidated.
There is no automatic fallback if the live file is unavailable. Input/output
collisions are rejected.

## Validation

Run the synthetic test suite from the project root:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Private validation exports and participant-level comparisons are not distributed.
Tests cover identity matching, scoring conventions, calendar boundaries,
follow-up handling, FRAT-up joins and cohort assembly. They do not establish
clinical validity of the scoring conventions.
