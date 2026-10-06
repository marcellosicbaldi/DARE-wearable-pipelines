# Changelog

## 0.1.0 — 2026-10-06

First consolidated release for the Bologna and Ravenna wearable
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
