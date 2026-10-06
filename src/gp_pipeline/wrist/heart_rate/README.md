# Heart Rate

This module contains the Empatica PPG heart-rate workflow migrated from
`legacy/hr/beliefppg.ipynb`.

The workflow:

- read `ppg.parquet` and `acc.parquet`
- split recordings at gaps in either sensor, using each configured sample frequency
- trim both streams to common coverage and keep continuous portions lasting at least 10 minutes
- run BeliefPPG on portions lasting at least 5 minutes
- save 2-second heart-rate estimates and uncertainty values

## Inputs

For each participant/visit, the runner expects:

- `input_root/<participant>/<visit>/Empatica/ppg.parquet`
- `input_root/<participant>/<visit>/Empatica/acc.parquet`

## Outputs

By default, outputs are written to:

- `output_root/<participant>/<visit>/Empatica/beliefppg/hr_belief.csv`

The CSV contains:

- `hr`
- `uncertainty`

with the BeliefPPG output timestamps stored as the CSV index.

## Run

```bash
gp-heart-rate --config configs/heart_rate.local.toml --participant 900001 --visit T0
```

The same workflow can be run interactively from:

- `notebooks/run_heart_rate_pipeline.ipynb`

Existing output is reused only when a completion record matches the inputs,
parameters and output hash. Empty or failed runs are retryable. Each stream
uses its own sampling frequency when checking duration and recording gaps.
