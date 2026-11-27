# Changelog

## Unreleased

- Declare the reviewed GP T0 profile in the standard GP example, with explicit
  timestamp recovery and inversion-only options. Retain compatibility API
  defaults and the Ravenna nonwear example; existing private TOML files require
  explicit activation. Document the fixed benchmark and its false-positive cost.
- Add a configurable joint wear report from saved McRoberts/Empatica intervals,
  with shared 168-hour windows, exact overlap, censoring QC and input hashes.
- Remove study-specific identifiers and recording dates from public research
  examples and reference-correction tooling. Keep corrections in private inputs.
- Correct gait step/turn counts and use participant height for DMO plausibility
  thresholds while retaining sensor height for stride-length calculation.
- Add the pinned Python GGIR 2023 implementation, optional gyro wear override,
  conservative timestamp-anchor quarantine, annotation UI and comparison tools.

- Add a T0-only full GP run profile with the selected nonwear algorithm and
  shared binary belt-inversion correction for gait and lying/TIB. Reverse
  vertical/ML in both triads while preserving PA; disable gait tilt alignment
  by default. Export walking-evidence QC and uncertain orientation coverage.

- Add the selected lower-back nonwear profile: 30-minute ACC windows on a
  one-minute grid, with a 30-second/50%-overlap gyro respiration override and
  final 15-minute minimum bout duration. Preserve compatibility API settings.
- Add an explicit elapsed protocol limit and timestamp-gap segmentation for the
  selected profile, matching the reviewed benchmark's observation window.
- Add a full GP lower-back batch launcher with frozen inventories, separate
  output directories, sequential recording workers, per-recording logs,
  checksum-verified resume and gait/TIB aggregation. Cohort execution remains a
  user action; preparing the code does not run participant recordings.

## 0.2.0 — 2026-10-06

- Rename the studies to DARE-FALLSPREDICT GP and DARE-FALLSPREDICT.
- Rename Python packages: `gp_pipeline` → `fallspredict_gp_pipeline`, and
  `ravenna_pipeline` → `fallspredict_pipeline`. Update Python imports and
  `python -m` invocations when upgrading from 0.1.0.
- Retain existing console command names, configuration filenames, input/output
  paths and cohort codes (`BO`/`RA`) for compatibility with existing analyses.
- Separate study display names from legacy clinical data folder names.

## 0.1.0 — 2026-10-06

First consolidated release for the DARE-FALLSPREDICT GP and DARE-FALLSPREDICT wearable
processing workflows.

- Shared `dare_wearables` core with separate cohort configuration and commands.
- Empatica EmbracePlus and GENEActiv sleep, circadian rhythm and activity processing;
  lower-back gait/posture, HR/HRV and clinical aggregation.
- Corrected timestamp alignment, gap/nonwear handling, zero-gait days and
  participant matching; explicit HRV interpolation reporting and rerun checks.
- Locked Python 3.11 environment, optional BeliefPPG heart-rate installation,
  synthetic regressions and GitHub distribution checks.
- MIT project license with retained third-party notices and method attribution.

Synthetic regression and model-loading checks do not establish clinical accuracy
or equivalence on private study recordings.
