# Manual lower-back nonwear annotation

This separate local app displays raw McRoberts acceleration above raw gyroscope,
with the self-reported lower-back removals from the baseline diary. It does not
run or modify nonwear detection, respiration, gait, time in bed, or existing results.
It needs no extra packages beyond the repository environment and a desktop browser.

## Start

From the repository root:

```sh
.venv/bin/python scripts/annotate_lower_back_nonwear.py
```

The browser opens automatically. Keep the terminal open; stop the app with Ctrl+C.
If the browser does not open, use the local address printed in the terminal.
An optional notebook launcher is in `notebooks/nonwear_annotation.ipynb`.

The private settings are in `configs/nonwear_annotation.local.toml`. For a new
installation, copy `configs/nonwear_annotation.example.toml`, set paths, and
choose the initial participant. Paths resolve relative to the configuration file.
The source data drive must be mounted. Other participants with baseline diaries
and raw recordings appear in the participant selector.

The first load decodes the OMX and builds an exact float64/raw-timestamp cache;
it may take a minute and uses several GB of disk per seven-day recording. Later
loads reuse the cache. Cache files can be deleted when the app is stopped;
**keep the separate labels folder**. Preload without opening a browser with:

```sh
.venv/bin/python scripts/annotate_lower_back_nonwear.py --prepare-only
```

## Review an interval

1. Enter your reviewer name. The starting participant comes from the private configuration.
2. Select an interval. Orange shading is the unchanged diary report; teal shading
   and draggable edges are your proposed interval, shown on both plots.
3. Drag the start/end edges, or enter exact dates and times and click **Apply times**.
   Both endpoints snap to retained raw sample boundaries. Every edit requires confirmation again.
4. Click **Confirm interval**, then **Save**. Saving preserves pending drafts too,
   but only explicitly confirmed intervals enter the labeled export.
5. Move to the next interval or participant. Switching participants saves the current
   drafts first. The next launch restores your saved work. Closing with unsaved
   changes triggers the browser's usual warning.

**Reject** excludes an incorrect diary proposal while retaining its audit record.
**+ Nonwear** or **+ Wear**, followed by dragging across either plot, adds an
interval absent from the diary. You may label removals of any duration; the
production algorithm's 15-minute cutoff is not imposed on manual annotations.
**Undo** restores the preceding edit; save again to keep the restored version.
The label selector can change an interval between wear and nonwear.

## Plot controls

- Drag away from a boundary to pan both plots in time.
- Scroll to zoom around the pointer, or use **Zoom +**, **Zoom −**, or **Drag to zoom**.
- **Shift + scroll** changes the vertical scale of the plot under the pointer.
  Double-click that plot or click **Fit Y** to reset the scale.
- Toggle individual axes using their checkboxes.
- **Focus interval** restores the selected episode and the chosen surrounding context.
  **Whole week** shows the complete available protocol window.

At broad zoom levels, min/max display envelopes retain motion peaks. At close
zoom (up to 30,000 samples in the view, five minutes at 100 Hz), every original
sample is plotted. The resolution is always stated under the plots. Raw values
use the production reader's units/orientation, before autocalibration, with no
signal filtering. All times retain the recording's stored local clock; the
browser does not convert them to another timezone.

## Saved dataset

For each participant and visit, the configured labels folder contains:

- `annotations.json`: authoritative saved session, including pending, confirmed
  and rejected entries, the original diary, source identities, reviewer, and sample bounds.
- `reviewed_intervals.csv`: confirmed wear/nonwear only, regenerated from the saved
  session. **Export confirmed labels (CSV)** downloads the same table.
- `history/`: previous saved revisions, allowing recovery from later mistakes.

Intervals use half-open sample bounds `[start_sample, stop_sample)` and exclusive
timestamp ends. Timestamp gaps are hatched and excluded from the CSV: an annotation
crossing a gap exports as separate observed portions. Boundaries cannot be placed
inside missing time. Conflicting confirmed intervals are rejected; adjacent intervals
are allowed. No uncertainty fields or boundary ranges are used.

Unreviewed time, blank diary days and rejected entries do not establish wear.
The two training labels are **wear** and **nonwear**, each with one exact start
and end. The annotation UI deliberately shows no algorithm predictions.

## Diary conventions

Import uses `patient_id`, `baseline_arm_1`, `encounter_ds` and the `no_sensor`
count and four `no_sensor_time_*` start/end pairs, with `_g2` through `_g7`
suffixes. These are the waist/lower-back fields, not the wrist fields.
Daytime periods use the diary date. If the end clock precedes the start clock,
the start belongs to the previous date and the end to the diary date. Blank
counts mean no reported episodes under the agreed study rule. Missing/invalid
endpoints and unexpected populated slots are excluded and listed in the app.
Original reports are preserved without shifting their dates to match signals.

## Verification without participant data

```sh
.venv/bin/python scripts/annotate_lower_back_nonwear.py --demo
.venv/bin/python -m unittest discover -s tests -p 'test_nonwear_annotation.py' -v
```

The demo uses invented signals and a temporary output folder. Its annotations
are discarded when that demo server stops. The real app retains its saved labels.
