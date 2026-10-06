# DARE-wearable-pipelines

[![Distribution checks](https://github.com/marcellosicbaldi/DARE-wearable-pipelines/actions/workflows/ci.yml/badge.svg)](https://github.com/marcellosicbaldi/DARE-wearable-pipelines/actions/workflows/ci.yml)

Research wearable workflows for the Bologna and Ravenna studies:

- Empatica wrist sleep, circadian rhythm, activity intensity, heart rate and HRV
- GENEActiv wrist sleep, circadian rhythm and activity intensity
- Lower-back IMU gait and time-in-bed processing
- REDCap, FRAT-up import and combined cohort aggregation

This repository contains the sanitized public workflows, with regression cases and fixes
for the reviewed processing errors and an extracted shared processing core.
Installation uses a locked environment, and GitHub Actions validates the
distribution. The project is MIT-licensed, with incorporated-source notices
preserved alongside the code.
The existing `gp_pipeline` / `ravenna_pipeline` console commands are retained.

## Contents

```text
configs/       Generic example configurations; no private study settings
docs/          Processing and data-contract documentation
notebooks/     Four unexecuted entry-point examples
scripts/       Publication-content checks
src/           Shared core and Bologna/Ravenna adapters
tests/         Synthetic regression tests
```

Study data, saved notebook results, exploratory/legacy notebooks, local
diagnostic reports, editor state, third-party reference PDFs and private
exclusion lists are not included. See [publication preparation](docs/publication.md).

## Shared architecture

Keep both cohorts in one repository and branch. Cohort differences are explicit
configuration and sensor adapters; separate branches would duplicate processing
fixes and make the two implementations drift.

```text
src/dare_wearables/     Sensor readers, preprocessing, algorithms, shared runners,
                       clinical processing and aggregation calculations
src/gp_pipeline/        Bologna configuration, commands and default paths
src/ravenna_pipeline/   Ravenna configuration, commands and GENEActiv orchestration
```

Both adapters depend on `dare_wearables`; the core imports neither cohort.
Empatica and GENEActiv preprocessing retain their device-specific loading and
nonwear rules, then feed the same `run_wrist_from_preprocessed` stage sequence.
The shared lower-back runner accepts either cohort's configuration. Aggregation
adapters supply sensor folders and gait QC policy explicitly. Existing import
paths forward to the core or use small wrappers that preserve their defaults.

For new Python integrations, use the neutral APIs in
`dare_wearables.wrist.sleep.pipeline` (`run_wrist_from_empatica`,
`run_wrist_from_empatica_silver`, `run_wrist_from_preprocessed`) and
`dare_wearables.lower_back.pipeline` (`run_from_config`). Shared aggregation
functions require explicit roots and sensor subdirectories. See the
[architecture diagram](docs/pipeline_diagram.md) and
[extraction details](docs/publication.md#shared-core-extraction).

## Setup and private configuration

The supported release environment is **Python 3.11 on Apple Silicon macOS or
Linux x86_64**. Other Python versions and platforms have not been validated.
The lockfile pins all resolved dependencies and their artifact hashes; direct
scientific dependencies are also pinned in package metadata.

Download a [GitHub release](https://github.com/marcellosicbaldi/DARE-wearable-pipelines/releases), or clone the repository and check out the desired release tag. Enter the repository directory,
and install [uv](https://docs.astral.sh/uv/getting-started/installation/) version
`0.12.23`. Then:

```bash
uv sync --locked
cp configs/empatica_sleep.example.toml configs/empatica_sleep.local.toml
```

This creates an isolated `.venv` and installs all three packages. For BeliefPPG
heart-rate inference, install the additional TensorFlow dependencies:

```bash
uv sync --locked --extra heart-rate
```

Activate the environment before running commands (`source .venv/bin/activate`
on macOS/Linux), or prefix each command with `uv run --no-sync`.
`uv sync` without the extra removes optional heart-rate dependencies; use
`uv run --no-sync` to preserve your chosen environment. HRV does not require
this extra.

For a plain pip installation from the downloaded source, use Python 3.11 and
`python -m pip install .` (or `'.[heart-rate]'`). This pins direct dependencies
but does not reproduce every transitive version from `uv.lock`; the locked
workflow above is the reference installation. See
[GitHub installation and releases](docs/installation_and_releases.md) for wheel
installation, dependency updates and release checks. The project is distributed
through GitHub only; it is not published on PyPI.

Copy each required `configs/*.example.toml` to the corresponding `*.local.toml`,
then edit that local file with your private paths and participant selection.
Local configuration files are ignored by Git and excluded from distributions.
Examples use the fictional identifier `900001`; no example data is supplied.

Bologna data paths currently require absolute paths, with a leading `~`
expanded to your home directory. Ravenna config-relative paths are resolved
from the configuration file's folder. Relative `--config` paths resolve from
your current working directory, including after wheel installation. Generic aggregation defaults use
`~/dare-data`; override them with the command's path arguments as needed.

## Run

After preparing the matching local configuration:

```bash
gp-empatica-sleep --config configs/empatica_sleep.local.toml --participant 900001 --combined
ravenna-sleep --config configs/ravenna_sleep.local.toml --participant 900001 --combined
gp-lower-back --config configs/lower_back.local.toml --participant 900001
ravenna-lower-back --config configs/ravenna_lower_back.local.toml --participant 900001
gp-heart-rate --config configs/heart_rate.local.toml --participant 900001 --visit T0
gp-hrv --config configs/hrv.local.toml --participant 900001
```

Equivalent examples are in `notebooks/`. Keep notebook outputs cleared before
committing. Inspect participant selection before any batch run.

## Input and output contracts

Empatica sleep expects `silver/<participant>/<visit>/Empatica/acc.parquet` and
`temp.parquet`. HR and HRV expect `ppg.parquet` and `acc.parquet`. HRV also uses
sleep windows from `Empatica/sleep_circadian/sleep_output_all_guiders.csv`.
Optional diary, recruitment tracker and lower-back TIB inputs are configured
privately. GENEActiv expects a participant/visit `.bin` recording; see the
[Ravenna input conventions](src/ravenna_pipeline/README.md).

Lower-back processing expects `bronze/<participant>/<visit>/<sensor>/*.OMX`
and requires REDCap metadata for gait processing. Its outputs include daily
gait, walking bouts, nonwear and time-in-bed tables. Wrist outputs live under
the participant/visit/sensor folder, with `sleep_circadian`, `beliefppg`, `hrv`
and `nocturnal_bursts` subdirectories as selected by the configuration.

All generated outputs and clinical/sensor inputs remain private. Omitting an
ID column from an export is not an anonymization procedure.

## Aggregation and private exclusions

```bash
gp-aggregate-all --silver-root "$HOME/dare-data/silver" --visit T0
gp-aggregate-redcap --redcap-csv "$HOME/dare-data/REDCap/Bologna/fallspredict_data.csv" --output-dir outputs/redcap
gp-aggregate-cohorts --exclusions-config configs/analysis_exclusions.local.toml --sleep-method mean
```

The combined-cohort command requires either a private `--exclusions-config`
or an explicit `--no-manual-exclusions`. This prevents a missing private rule
file from silently changing a study's exclusion decisions. The distributable
`analysis_exclusions.example.toml` has empty lists; enter your reviewed study
decisions in the ignored local copy. Automatic minimum-data masks and column
selection apply in both modes.

See [combined cohort assembly](docs/combined_cohorts.md),
[REDCap processing](docs/redcap_processing.md), and
[aggregation thresholds](docs/aggregation_thresholds_and_nonwear.md).

## Checks

```bash
uv run --no-sync python -m unittest discover -s tests -v
uv run --no-sync python scripts/check_publication.py
uv run --no-sync python scripts/check_install.py
```

The publication check inspects Git-visible working files and staged content
for disallowed files, local paths, common credential patterns, notebook
execution state and nonempty example exclusion lists. It reports locations,
not sensitive values. It does not inspect Git history or certify arbitrary
text as anonymous; keep future additions limited to reviewed code, prose and
synthetic examples.


## Licensing and attribution

This project is licensed under the [MIT License](LICENSE), copyright (c) 2026
Marcello Sicbaldi. Third-party code retains its original notices; see
[third-party notices and provenance](THIRD_PARTY_NOTICES.md) and `licenses/`.
