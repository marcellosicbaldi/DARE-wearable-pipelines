# Lower-back profile, evidence and promotion

The reviewed GP T0 profile addresses [issue #1](https://github.com/marcellosicbaldi/DARE-wearable-pipelines/issues/1):
very quiet worn periods can satisfy acceleration-only nonwear thresholds.
Respiration-like gyro activity supplies positive evidence for restoring wear.
Absence of that evidence never creates nonwear.

## Configuration contract

`configs/lower_back.example.toml` and
`configs/lower_back_t0_inversion.example.toml` explicitly select the same
processing parameters for GP T0:

| Stage | Selected configuration |
| --- | --- |
| Protocol | First 168 elapsed hours, including recording gaps |
| ACC detector | Python GGIR 2023 port, pinned to GGIR 3.3-9 |
| ACC grid | 30-minute forward windows, one-minute decisions |
| ACC thresholds | SD <13 mg **and** range <50 mg on at least two axes |
| Continuity | Retain GGIR gap/start rules; disable recording-end extension and the final-day download assumption |
| Gyro screen | Detrended Hann spectra in 30-second windows, 50% overlap, any passing axis, all hours |
| Respiration band | Strictly 10 < BR < 30 brpm |
| Spectral thresholds | RMS >=0.01 degrees/s; band fraction >=0.5; peak concentration >=0.5; peak/background >=6 dB |
| Residual nonwear | Keep bouts >=15 observed minutes after respiration and gap splitting |
| Timestamp policy | Quarantine only verified bad-anchor patterns; exclude missing/time-invalid spans |
| Mounting | Identity or 180 degrees around PA, reversing vertical/ML in both ACC and gyro |
| Tilt | No tilt alignment in the production run |

The issue's original proposal mentioned 9–30 brpm and both 30/60-second gyro
windows. The selected profile uses strict 10–30 brpm and 30-second windows.
The 15-minute output-duration rule does not establish sensitivity to isolated
15-minute removals: the ACC evidence window is still 30 minutes.

Library defaults retain 60-minute ACC windows, 15-minute decisions, strict
timestamps and disabled respiration for compatibility. The Ravenna example
retains those nonwear settings. Existing private configurations are never
rewritten by an upgrade. The shared runner now applies the binary inversion
correction by default; explicitly set `orientation.method = "none"` to disable
it. Reproduction of the old all-axis reflection uses the comparison tools.

## Descriptive benchmark

Twenty complete T0 recordings were manually reviewed with the independent
annotation app. Observed time outside confirmed nonwear was treated as wear
following the reviewer's confirmation of full-recording review. Twenty-seven
candidate configurations were declared before scoring and evaluated on all
20 participants, without a held-out split or subsequent parameter tuning.

The final reference includes one reviewer-confirmed missing episode, for
63 episodes in total. A gap-spanning annotation remains one logical episode;
missing samples are excluded from scoring. Timestamp-only gap ledgers never
generate synthetic ACC or gyro samples. One-to-one episode matching requires
at least 50% observed-sample intersection over union.

| Configuration | Mean participant F1 | Episodes matched / 63 | False nonwear minutes / observed day |
| --- | ---: | ---: | ---: |
| Benchmark baseline: 60/15 ACC, 60-second gyro | 0.270 | 9 | 9.8 |
| Selected: 30/1 ACC, 30-second gyro | 0.406 | 34 | 43.4 |
| 30/1 ACC, 60-second gyro, no additional gap/edge rules | 0.405 | 34 | 41.1 |
| 15/1 ACC, 60-second gyro, no additional gap/edge rules | 0.391 | 50 | 71.8 |

All four rows retain the 15-minute minimum final nonwear duration. Window and
step values are minutes. The benchmark baseline is the GGIR 2023 comparator,
not the original `main` branch's custom 2013-style implementation and not a
comparison of final October/November datasets.

The two 30/1 candidates were effectively tied; the 30-second gyro profile was
selected. Increased episode recovery comes with substantially more false
nonwear. These are descriptive results on the reviewed sample, not independent
validation or proof of generalization. Pooled duration scores can be dominated
by recordings with long removals; participant-mean and episode scores are both
needed. Private annotations, detailed predictions and participant reports stay
outside the repository.

## Orientation and gait interpretation

The historical operation reversed all three axes. The reviewed correction is
a proper 180-degree belt rotation that preserves PA. Walking evidence during
continuous final wear determines the mounting state; lying and nonwear do not
supply orientation votes. Insufficient or conflicting evidence leaves the
signals unchanged and produces QC flags. Gait and lying/TIB consume the same
corrected signals. Orientation does not change the nonwear mask.

Acceleration magnitude, and therefore initial Ionescu gait detection on the
same input boundaries, is invariant to this rotation. Signed extraction can
still change contacts, laterality, retained strides and final walking bouts.
Earlier experimental comparisons included continuous tilt alignment; their
numerical gait/TIB differences must not be presented as the effects of the
final inversion-only production profile.

Gait exports now count contacts and turns explicitly and retain `n_strides`
separately. Plausibility thresholds use participant height; sensor height is
reserved for stride length. Changes in reported steps therefore include a
definition correction as well as any extraction effect.

## Verification and completed cohort run

Synthetic regressions exercise GGIR parity, strict threshold boundaries,
timestamp quarantine, segmentation, minimum bout duration, shared inversion,
contact counts and resumable completion manifests. Optional R checks use the
pinned upstream functions; production detection needs no R runtime or downloads.
During investigation, the Python port also reproduced an earlier R-backed
assessment on ten randomly selected recordings. This historical equivalence
check does not validate the scientific accuracy of the selected adaptation.

The final inversion-only run completed all 197 GP T0 recordings. Its frozen
source inventory contains 273 Python files, all matching PR commit `9453745`
byte for byte. A subsequent review fix passes `protocol_hours` through serial
and parallel nonwear-only assessment workers and includes the requested limit
in nonwear provenance/completion signatures. The full-run launcher already
passed the elapsed limit correctly; its signal processing is unchanged by this
fix. Current sources therefore no longer match every historical source hash.
The completed run and its manifests remain attached to the frozen revision:
use that snapshot to verify/resume it, and explicitly review the later source
differences before promoting its results under a newer release. Never rewrite
the old manifests to claim that they were produced by the newer code.

The cohort's completion status establishes execution, not accuracy. Output
hashes, environment, metadata and scientific QC must still be checked before
promoting those outputs.

## Reusable research tooling

The annotation, comparison and inspection scripts take private paths and
participant choices from arguments or ignored local configurations. Notebook
examples use invented identifiers. Reference-gap export applies to any
gap-spanning annotation.

To add a reviewer-confirmed missing reference episode without rerunning a
detector, supply `scripts/rescore_nonwear_benchmark.py` with a private JSON file:

```json
{
  "subject": "900001",
  "visit": "T0",
  "annotation_id": "reviewed_missing_episode",
  "requested_start": "2024-01-01T08:00:00",
  "requested_end": "2024-01-01T08:30:00",
  "reason": "Reviewer-confirmed missing removal",
  "authority": "Reviewer and review date"
}
```

```bash
.venv/bin/python scripts/rescore_nonwear_benchmark.py /absolute/path/to/benchmark_run \
  --correction /absolute/path/to/reference_correction.local.json
```

The script requires the parent's frozen detector/scoring sources, verifies
saved predictions, creates a separate corrected result and preserves original
annotations and parent reports. Research-tool cleanup changes research-script
hashes: use an experiment's frozen tool revision to resume that experiment.
Never edit old manifests to relabel their provenance.

## Remaining promotion and extraction work

1. Review the PR, pass CI on the final commit, merge and create a versioned release.
2. Verify the completed run against its frozen sources and environment, then
   review any differences from the released code before reusing its outputs.
3. Use the new gait and combined valid TIB outputs; refresh TIB-dependent wrist
   sleep and any HR/HRV outputs consuming those changed sleep windows.
4. Assemble an isolated set of approved GP inputs and rebuild the GP overall
   export. `overall_T0_sleep_mean.csv` takes precedence over individual gait
   and sleep files in final cohort assembly.
5. Build a new extraction directory using explicit GP/Ravenna sensor sources,
   clinical inputs, FRAT-up inputs, exclusions and the existing mean-sleep policy.
   Record the source versions and release. Compare rows, missingness, valid-day
   counts and measurements against the saved previous extraction.

Merging code does not move data, activate an existing private configuration,
close a private processing run, or refresh a previously exported dataset.
