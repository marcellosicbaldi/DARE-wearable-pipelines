# Lower-back timestamp quality

The McRoberts reader defaults to `strict`: duplicate or backward timestamps
reject the recording. A verified recording-side failure can be handled with an
explicit option in either cohort's lower-back TOML:

```toml
[lower_back.timestamps]
policy = "quarantine_bad_anchors"
```

This policy never rewrites raw files, sorts samples, shifts timestamps or inserts
replacement observations. It excludes packets tied to a narrowly recognized bad
time anchor before aligning acceleration and gyro by timestamp. Excluded time is
unknown; it contributes no wear or nonwear samples. Nonwear percentages use
observed samples; TIB coverage separately reports unobserved wall time.

The recognized pattern is a short, interior group of 2–40 packets sharing one
anchor at the end of a minute, with a matched forward/backward minute- or
hour-scale excursion, continuous packet counters, valid OMX headers, 100 Hz
sampling and valid checksums throughout the group and its two bracketing packets.
The nominal affected support must be at most 32 seconds. Retained packet times
must be strictly increasing. An unrecognized anomaly, overlapping restart, bad
checksum or ambiguous alignment still rejects the recording. These fixed bounds
identify the verified failure pattern; they are not nonwear thresholds.

Each stream is handled independently. The existing quarter-sample-period
alignment tolerance remains in force; gyro samples cannot be reused. When only
one stream has a bad anchor, observations lacking a valid counterpart are also
excluded by alignment. The QC record distinguishes packet exclusions from
unmatched observations. Counter resets and positive time gaps are retained.

For a recovered recording, GGIR and respiration run within continuous observed
segments and restart at timestamp gaps and recording discontinuities. GGIR uses
the same matched thresholds and postprocessing within each segment. Its final
partial 15-minute epoch in each segment stays outside the decisions, as in the
unsegmented profile; the provenance exports these unclassified tails. Gait also
respects the segment boundaries. The TIB runner prevents missing seconds from
being filled as wear or exported as nonwear and reports separate observed-time
coverage. Sleep/TIB windows may span a gap, but gap seconds provide no evidence
of wear or posture.

A file that needs no quarantine or restart handling follows the original
nonwear calculation. Enabling this option does not change its observed samples,
GGIR decisions or respiration features.

The production runner writes `time_quality/`; the nonwear-only assessment also
includes the following in its per-recording `nonwear/` folder:

- `timestamp_quality.json`: policy, excluded packet/sample counts, retained-time
  contract, per-stream events, restarts and alignment diagnostics. Events cover
  the entire input; the `analysis` section gives the capped processing bounds.
- `time_invalid_intervals.csv`: half-open gaps between observed paired samples,
  clipped to the analyzed recording, with reasons for quarantine, recording
  discontinuity or other missing/unaligned observations. Durations represent
  unobserved wall time, separately from nominal excluded sample duration.
- `recording_discontinuities.csv`: counter changes and their recorded time gaps.

Respiration summaries and daily tables report `time_invalid_hours` separately.
The TIB output `wear_time_quality_1hz.csv` has an explicit `time_valid` field and
empty wear values for unobserved seconds. Nonwear bouts include observed seconds
only. Recovery settings and module hashes are part of assessment completion
signatures; changing them requires a new run, preventing stale output reuse.

Validate recovery in a new output root before recomputing dependent gait, TIB
and cohort aggregates. Known invalid raw samples remain available in the original
OMX for any later clock reconstruction validated against an independent export.
