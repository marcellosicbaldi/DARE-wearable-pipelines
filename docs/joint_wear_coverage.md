# Joint McRoberts and Empatica wear coverage

`scripts/summarize_joint_wear.py` reconstructs estimated T0 wear coverage from
saved reports. It reads no raw sensor recordings and does not rerun either
nonwear detector. Every input path is supplied explicitly:

```bash
.venv/bin/python scripts/summarize_joint_wear.py \
  --empatica-dir /absolute/path/to/empatica_nonwear_report \
  --mcroberts-run /absolute/path/to/completed_lower_back_run \
  --output /absolute/path/to/new_joint_wear_report
```

The output directory must not exist and must be separate from both source
trees. Add `--no-plots` to omit PNG/SVG figures. The script creates subject and
daily summaries, wear intervals, four-state coverage intervals, cohort summary,
cumulative thresholds, mutually exclusive coverage bands, methods, HTML report,
validation and SHA-256 input provenance. `workbook_data.json` contains the same
tables for optional workbook authoring; this command does not generate a workbook.

## Inputs and window

The Empatica directory supplies `subject_summary.csv` and `intervals.csv` from
the existing nonwear report. The completed lower-back run supplies
`run_status.json` and each recording's `nonwear_summary.json`,
`timestamp_quality.json`, `final_nonwear_intervals.csv` and
`time_invalid_intervals.csv`. Both sources must have matching, unique T0
subject/visit keys and completed lower-back status. Inconsistent sources fail;
subjects are not silently dropped. Identifiers remain text.

All three measures use each subject's Empatica
`[recording_start, protocol_end)` interval and the same 168-hour denominator:

| Measure | Definition |
| --- | --- |
| McRoberts | Processed observation span minus time-invalid intervals and final nonwear |
| Empatica | Complement of detected nonwear, recording gaps, merge padding and missing recording |
| Both simultaneously | Exact timestamp intersection of the two wear interval sets |

Intervals are half open. A McRoberts sample has support through one sample
period after its timestamp, clipped to the original elapsed 168-hour cap.
Posture/TIB's separate nonwear filling is excluded from wear coverage.
Device timestamps are not shifted to make the recordings align.

Daily summaries contain seven successive 24-hour periods from the common
start, rather than calendar days. Threshold counts use cumulative `>=` cutoffs
on unrounded percentages. Coverage bands are mutually exclusive. No clinical
exclusions or minimum valid-day masks are applied.

## Interpretation and checks

These are algorithm-defined estimates of wear coverage, not confirmed physical
wear or physiological signal quality. Empatica retains the source report's
handling of short gaps and DETACH-unassessed time. Source totals and the new
interval calculations must reconcile; those source limitations are carried in
the subject QC fields and methods notes.

McRoberts time outside processed coverage remains unavailable. If its own
168-hour processing cap ends before the shared window, the omitted tail is
flagged and contributes no assumed wear. The report supplies upper bounds if
that tail were worn, and tests whether those bounds could change threshold
counts. When retained-source counts are unavailable, cap status is explicitly
unknown; no cap-censoring upper bound is inferred.

The four states `both`, `mcroberts_only`, `empatica_only` and `neither` must
partition exactly 168 hours. An independent endpoint sweep verifies the
intersection; daily totals must reconcile with subject totals. Neither means
neither device has estimated wear coverage, not confirmed simultaneous removal.
Malformed intervals, mismatched cohorts and inconsistent source durations are
rejected. Source hashes are checked again before publishing the report.

Stored timestamps must be timezone-naive and mutually comparable. The script
retains their existing clock interpretation and performs no timezone/DST
conversion or additional clock synchronization.

The initial private analysis covered 197 matched GP T0 subjects. Median coverage
was 96.6% for McRoberts, 94.2% for Empatica and 89.8% simultaneously. These
aggregate findings describe that analysis; the script computes sample size,
summary values and figure limits from the supplied sources.
