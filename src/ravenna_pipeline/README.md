# Ravenna Pipeline

This package is the Ravenna configuration and command adapter over the shared
`dare_wearables` core. It is a sibling of the Bologna `gp_pipeline` adapter.

The Ravenna wrist entry point feeds the shared sleep, circadian and activity
stages after sensor-specific preprocessing:

- GENEActiv `.bin` loading via `dare_wearables.wrist.data_io.geneactiv`
- Van Hees / GGIR non-wear detection via `nonwear_method = "vanhees2013"`

## Run

From the project root, after editing `configs/ravenna_sleep.local.toml`:

```bash
python -m ravenna_pipeline.cli.ravenna_sleep --config configs/ravenna_sleep.local.toml --participant 900001 --combined
```

You can also run the notebook-style wrapper script directly:

```bash
python src/ravenna_pipeline/main/wrist_pipeline.py --participant 900001
```

If the project is installed with `pip install -e .`, the equivalent console command is:

```bash
ravenna-sleep --config configs/ravenna_sleep.local.toml --participant 900001 --combined
```

Example configurations are in the root `configs/` directory. Copy them to
ignored `*.local.toml` files and configure your own private paths.

## Expected GENEActiv Input

Set `input_root` in `configs/ravenna_sleep.local.toml`. The loader will search common layouts such as:

```text
input_root/<participant>/<visit>/GENEActiv/*.bin
input_root/<participant>/<visit>/*.bin
input_root/<participant>/GENEActiv/<visit>/*.bin
```

By default, `configs/ravenna_sleep.local.toml` also requires the `.bin` filename to
start with the participant ID, for example:

```text
900001/T0/GENEActiv/900001_left_wrist_recording.bin
```

If automatic discovery finds zero or multiple `.bin` files, set `geneactiv_file_path` in the TOML. Relative `geneactiv_file_path` values are resolved from `input_root`.

Outputs are written to:

```text
output_root/<participant>/<visit>/GENEActiv/sleep_circadian/
```

The shared lower-back workflow uses Ravenna configuration and can be run with:

```bash
python -m ravenna_pipeline.cli.lower_back --config configs/ravenna_lower_back.local.toml --participant 900001
```

Or with the notebook-style wrapper script:

```bash
python src/ravenna_pipeline/main/lower_back_pipeline.py --participant 900001
```

## Aggregate Outputs

Ravenna aggregation reads from:

```text
~/dare-data/ravenna/silver/<subject>/<visit>/
```

Available aggregation commands:

```bash
python -m ravenna_pipeline.aggregation.sleep
python -m ravenna_pipeline.aggregation.gait
python -m ravenna_pipeline.aggregation.activity_intensity
python -m ravenna_pipeline.aggregation.overall
```

Outputs are written by default to:

```text
~/dare-data/ravenna/silver/aggregation/
```
