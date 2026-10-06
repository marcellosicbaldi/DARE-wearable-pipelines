# Changelog

## 0.2.0 — 2026-10-06

- Rename the studies to DARE-FALLSPREDICT GP and DARE-FALLSPREDICT.
- Rename Python packages: `gp_pipeline` → `fallspredict_gp_pipeline`, and
  `ravenna_pipeline` → `fallspredict_pipeline`. Update Python imports and
  `python -m` invocations when upgrading from 0.1.0.
- Retain existing console command names, configuration filenames, input/output
  paths and cohort codes (`BO`/`RA`) for compatibility with existing analyses.
- Separate study display names from legacy REDCap folder names.

## 0.1.0 — 2026-10-06

First consolidated release for the DARE-FALLSPREDICT GP and DARE-FALLSPREDICT wearable
processing workflows.

- Shared `dare_wearables` core with separate cohort configuration and commands.
- Empatica and GENEActiv sleep, circadian rhythm and activity processing;
  lower-back gait/posture, HR/HRV and clinical aggregation.
- Corrected timestamp alignment, gap/nonwear handling, zero-gait days and
  participant matching; explicit HRV interpolation reporting and rerun checks.
- Locked Python 3.11 environment, optional BeliefPPG heart-rate installation,
  synthetic regressions and GitHub distribution checks.
- MIT project license with retained third-party notices and method attribution.

See `docs/publication.md` for changes that require reprocessing existing results.
Synthetic regression and model-loading checks do not establish clinical accuracy
or equivalence on private study recordings.
