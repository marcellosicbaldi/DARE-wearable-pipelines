# Publication preparation

This directory is a separate publication candidate created on 2026-10-05.
The original working analysis directory remains the private operational copy.
No study data or previous Git history was imported, and no GitHub repository
has been created or pushed by this preparation step.

## Included and omitted

Included: the shared core and both cohort adapters, synthetic tests, method documentation,
generic example configurations and four clean notebook entry points.

Omitted: generated CSVs and parquet files at every depth, notebook outputs,
exploratory and legacy directories, local debugging reports, reference PDFs,
editor/history files, real exclusion identifiers and institution-specific
workstation/network paths. Private validation results were removed from the
public documentation while method definitions were retained.

The original settings and notebooks remain available in the private working
directory. Its ignored `outputs/publication_preparation_20261005/` folder
contains the exact extracted exclusion configuration and the preparation
audit. These files are outside this publication tree.

## Using private study rules

Store real exclusion decisions in an external private TOML file or in the
ignored `configs/analysis_exclusions.local.toml`. Pass its path explicitly:

```bash
gp-aggregate-cohorts --exclusions-config /path/to/private/analysis_exclusions.local.toml
```

For runs intentionally without participant-specific exclusions, pass
`--no-manual-exclusions`. Python callers make the same choice with
`exclusions_config=...` or `no_manual_exclusions=True`. Direct callers of
`apply_analysis_exclusions` pass an `AnalysisExclusions` instance explicitly.
No implicit lookup of the old workspace or private study lists occurs.

The TOML must contain only `[manual_exclusions]` with both `rmssd_sdnn_ids`
and `sleep_circadian_ids`, each an array of quoted numeric identifiers.
Duplicate normalized IDs and invalid structures are rejected. Existing
Bologna-T0 HRV and Ravenna-T0 sleep/circadian/activity masking semantics are
preserved; the data source for the identifiers has changed.

## Preventing accidental reintroduction

Run `python scripts/check_publication.py` before committing or publishing.
Both working files and the Git index are checked, including force-added
ignored files. Data files, local configs and unreviewed binary assets are
rejected. Tests generate data in temporary directories; no participant
fixtures are distributed. Keep executed notebooks private or clear their
outputs, execution counts, attachments and execution metadata before adding.

Package discovery is restricted to `dare_wearables`, `gp_pipeline` and
`ravenna_pipeline`, with implicit
namespace discovery and automatic package-data inclusion disabled. Source
distribution rules exclude local configuration and study file formats.
Build artifacts should also be checked before release.

## Correctness work

The second preparation step fixes the reviewed processing failures in both
packages. Regression inputs are generated synthetically in temporary folders.

| Failure | Corrected behavior |
| --- | --- |
| Cohort depended on REDCap row position or an identifier prefix | Match the exact normalized participant identifier and baseline event; reject missing or conflicting diagnoses and heights. |
| Accelerometer and gyroscope packets were paired by position | Match timestamps within a quarter sample period. Report unmatched sample counts and reject missing streams, conflicting rates and ambiguous timestamps. |
| Gait included nonwear, and missing time was compressed | Keep actual elapsed timestamps and process each continuous wear segment independently. Aggregate the segment bouts together once per recording date. |
| Days without detected gait disappeared | Export zero amounts and undefined quality for successfully processed zero-gait days. Mark days without eligible continuous wear separately. |
| Walking-bout dates relied on nearest midnight offsets | Map segment sample bounds directly to the recording's timestamps, with an exclusive end. |
| Failed or skipped reruns could leave old tables | Invalidate owned stage tables before rerunning; remove partial results on exceptions. Gait errors propagate and leave an explicit status table. |
| Wrist processing missed PPG-only gaps | Intersect continuous intervals from both streams and check each configured sampling frequency. |
| HR/HRV reused outputs merely because files existed | Require a completion record covering parameters, input paths/sizes/modification times and hashes of every expected output. Empty runs remain retryable. |
| Sleep joins failed on different datetime resolutions | Cast interval boundaries to the epoch timestamp type before joining. |

HRV retains interpolation as requested. Each window exports
`contains_interpolated_beats`, `n_interpolated_beats` and
`interpolated_fraction`, counting rejected/artifact or out-of-range intervals
that were filled. The fraction denominator is the number of retained intervals
in that window. No maximum fraction is imposed. The IBI parquet now has named
`ibi` (seconds) and `interpolated` (boolean) columns. The first peak no longer
receives an invented interval: intervals are indexed by their ending peak.
Beat detection and interpolation never bridge a recording gap. Burst detection
preserves timezones and accepts stationary signals without generating bursts.

HRV interprets naive timestamps as clock times in its configured timezone and
converts aware timestamps while preserving their instants. Ambiguous naive
times at daylight-saving transitions require explicit timezone information.
Default HR and HRV directories now agree with their aggregators: `beliefppg`
and `hrv`. Explicit `output_subdir` settings still select another directory.

These fixes change derived results. Reprocess sensor outputs and rebuild
aggregations before comparing cohorts. Existing HR/HRV outputs without the
new completion records are recomputed. A failed rerun removes that stage's
previous derived tables; source recordings and unrelated files are retained.

Regression coverage includes packet alignment and axes, exact participant
matching, wear/gap boundaries, missing recording dates, zero-gait averages,
rerun failures, sleep datetime units, per-stream sampling rates, interpolation
fractions, timezone preservation and completion-record invalidation. The
stationary gait and accelerometer-burst cases also exercise the real detectors.
Deterministic replacements cover the expensive beat and HR inference paths.
This verifies processing behavior; it does not establish clinical accuracy or
equivalence on the private study recordings.

## Shared-core extraction

The third step moves common processing into `src/dare_wearables`. Bologna and
Ravenna now share sensor readers, calibration/nonwear helpers, lower-back gait
and posture processing, sleep/circadian/activity algorithms, HR/HRV, REDCap
processing and aggregation calculations. The core has no imports of either
cohort package; both adapters can evolve their configuration independently.

The common wrist runner accepts a prepared mapping with `calibrated_df`,
`acc_df`, `temp_df`, `nonwear_df` and `info`. Empatica preprocessing and GENEActiv
preprocessing produce this contract before invoking the same sleep, circadian
and activity stages. File discovery and cohort defaults stay in the adapters.
Neutral core entry points coexist with legacy GP-named function aliases for
compatibility; there is only one implementation of each algorithm.

Existing commands, configuration classes, output paths and processing thresholds
are retained. Compatibility modules forward old algorithm imports to the core;
small wrappers preserve cohort-specific function defaults and positional argument
order. In particular, Bologna retains its legacy gait-hours fallback while
Ravenna requires measured nonwear minutes. Direct lower-back calls retain their
previous size defaults, separately from the configuration defaults.

New architecture regressions check dependency direction, imports without either
cohort available, legacy module sharing, sensor delegation, positional arguments,
cohort defaults and aggregation equivalence on synthetic inputs. The full suite
also exercises the correctness regressions from the previous step. This
extraction changes code ownership; it introduces no new scientific thresholds.
Validation on private recordings remains a separate scientific check.

Extraction validation on 2026-10-05: all 101 regression tests passed. Temporary
wheel and source builds included all three packages without prohibited data
files. Imports from the unpacked wheel and `--help` for all 18 console commands
passed, as did module execution for eight aggregation entry points. The
publication check passed for 298 candidate files and the available Git index.

## GitHub distribution packaging

The distribution is named `DARE-wearable-pipelines`; the import packages and
existing commands keep their names. Version `0.1.0` is a local candidate until
a matching GitHub tag and release are created. There is no PyPI publication
workflow, and package metadata includes `Private :: Do Not Upload` to prevent
accidental PyPI upload.

`uv.lock` records a Python 3.11 environment for Linux x86_64 and Apple Silicon
macOS. The reference interpreter is pinned in `.python-version`. The optional
`heart-rate` extra installs BeliefPPG and its compatible TensorFlow stack.
MobGap 1.0.0 uses scikit-learn 1.6.1, matching its bundled laterality model.
The model compatibility check treats a version mismatch as an error.

GitHub Actions checks publication contents, regression tests, installed commands,
real model loading, wheel/source metadata and source-to-wheel rebuilding.
Successful Linux jobs retain the reviewed wheel, source archive and SHA-256
checksums as downloadable artifacts. Releases are created deliberately from
those assets after review; the workflow does not publish them automatically.

Relative configuration paths now resolve against the invocation directory, so
installed commands no longer depend on the source checkout's location. The
cohorts' rules for paths inside configuration files are retained.

Validation on 2026-10-06 used a fresh Python 3.11.17 environment on Apple
Silicon macOS. All 104 synthetic regressions passed in the base environment
without BeliefPPG or TensorFlow. The optional environment passed real BeliefPPG
inference on 51 synthetic windows and loaded MobGap's model with version
warnings treated as errors. Both editable and wheel installations passed the
18 command checks outside the repository. Wheel/source metadata and archive
contents passed, and rebuilding the source archive produced identical wheel
file contents. Linux dependency resolution was checked with a dry run; execution
on Linux and hosted GitHub CI are still pending.

## Remaining release work

The maintainer authorized the MIT License on 2026-10-06, with Marcello Sicbaldi
as copyright holder. NeuroKit2 is credited for MSPTDfast as well as peak
correction. The GGIR inspiration and colleague contribution are documented in
`THIRD_PARTY_NOTICES.md`; existing credits and upstream MIT notices are retained.
The project license and third-party notices are required in wheel/source audits.

The GitHub repository, hosted CI run and first public release remain pending.
Follow [installation and release instructions](installation_and_releases.md).
