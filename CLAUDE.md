# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`pamflow` is a Kedro data pipeline for processing passive acoustic monitoring (PAM) recordings, built for the Sonópolis biodiversity monitoring project in Bogotá. It ingests raw WAV files plus a field-deployments spreadsheet and runs them through data preparation, quality control, species detection (BirdNET), acoustic indices, graphical soundscapes, and GBIF export pipelines.

## Setup

```bash
conda create -n pamenv python=3.11
conda activate pamenv
pip install -r requirements-mac.txt   # or requirements-win.txt on Windows
```

Local config (not committed) must be created before running anything:
- `conf/local/parameters.yml` — set `DEVICES_ROOT_DIRECTORY` to the folder containing per-sensor audio subdirectories.
- `conf/local/catalog.yml` — set the filepath for `field_deployments_sheet@pandas` (an Excel sheet with a `recorderID`/`deploymentID` column matching the audio subdirectory names).

## Common commands

```bash
kedro run                                  # run the default pipeline (see below)
kedro run --pipeline=<name>                # run a single named pipeline
kedro run --pipeline=<name> --node=<node>  # run a single node
kedro viz                                  # visualize the pipeline DAG (needs kedro-viz)

pytest                                     # run tests (config in pyproject.toml, coverage over src/pamflow)
pytest src/tests/pipelines/data_preparation/test_node_get_media_file.py  # single test file
pytest -k test_get_media_file_basic        # single test by name

ruff check src/                            # lint (rules: F, E, W, UP, I, PL — see pyproject.toml)
```

Recommended first-run order (each is a registered pipeline name):
`data_preparation` → `quality_control` → `species_detection` → `acoustic_indices` → `graphical_soundscape` → `export`

## Architecture

### Kedro project layout
- `src/pamflow/pipelines/<pipeline_name>/{nodes.py, pipeline.py, utils.py}` — one folder per pipeline. `nodes.py` holds the actual transform functions (pure-ish, testable in isolation); `pipeline.py` wires them into a `kedro.pipeline.Pipeline` by declaring `inputs`/`outputs` as catalog dataset names; `utils.py` (where present) holds helpers used only within that pipeline.
- `src/pamflow/pipeline_registry.py` — assembles pipelines into named combos (`__default__`, `pamflow`, and one entry per individual pipeline name used with `--pipeline=`). `data_science` is intentionally excluded from `__default__`/`pamflow` composite unless explicitly requested.
- `conf/base/catalog/<pipeline_name>.yml` — Kedro Data Catalog entries, split by pipeline. Dataset names follow `<entity>@<type>` convention (e.g. `media@pamDP`, `deployments_gbif@pandas`, `graph_plot@PartitionedImage`) — the suffix after `@` indicates the dataset type/format, not a Kedro namespace.
- `conf/base/parameters/<pipeline_name>.yml` — parameters injected as `params:<key>` node inputs.
- `conf/local/` — gitignored, user-specific overrides (paths, credentials); merged on top of `conf/base/` at runtime.
- `data/input/` and `data/output/` — gitignored except for `.gitkeep`/small samples; pipeline outputs land here by default, organized by pipeline name.

### Pipeline dependency chain
Pipelines pass data through the catalog rather than calling each other directly. The rough data flow:
1. `data_preparation`: scans `DEVICES_ROOT_DIRECTORY` + the field deployments Excel sheet → produces `media@pamDP` and `deployments@pamDP` (the two datasets nearly every downstream pipeline consumes).
2. `quality_control`, `graphical_soundscape`, `acoustic_indices`, `species_detection` all consume `media@pamDP`/`deployments@pamDP` independently and can run in any order relative to each other.
3. `species_detection` produces `observations@pamDP` and segment audio/spectrograms used for manual annotation.
4. `data_science` consumes manually-annotated data (`manual_annotations@PartitionedDataset`) to find detection thresholds and build train/test spectrogram datasets — this is why it's excluded from the default pipeline (it depends on a human-in-the-loop annotation step).
5. `export` reformats `media@pamDP`, `deployments@pamDP`, and thresholded observations into GBIF/Darwin Core-style CSVs.

### Custom Kedro datasets (`src/pamflow/datasets/`)
- `audio_dataset.py` (`SoundDataset`) — loads/saves a single WAV as a numpy array via `scikit-maad`.
- `audio_folder_dataset.py` / `image_folder_dataset.py` — load/save a folder-of-subfolders-of-files as a nested dict, used for per-deployment segment audio and spectrogram image collections.
- `datasets/pamDP/` — implements the "pamDP" (PAM Data Package) tabular format, modeled after Darwin Core. `CSVPamDP.py` is the shared base class (`Media`, `Deployments`, `Observations`, `TargetSpecies`, `FieldDeployments` subclass it) and enforces, on every load/save: exact expected column set, dtype coercion via a schema dict, required-column null checks, uniqueness constraints, optional categorical/enum constraints, and ISO 8601 date normalization. When adding a new pamDP-typed catalog entry, define its column/schema/required/unique dictionaries alongside the class the same way `deployments.py` does, then register it in the catalog as `type: pamflow.datasets.pamDP.<module>.<Class>`.

### Notebooks
`notebooks/` contains exploratory/one-off analysis scripts (not part of the Kedro pipeline) — soundscape comparisons, PANNs tagging analysis, manual-validation workflows, EDA. These are run standalone, not via `kedro run`.

### Docs
`docs/source/` builds Sphinx/MyST docs (see `.readthedocs.yaml`); published docs are at pamflow.readthedocs.io. This repo's README defers general pamflow documentation there and keeps only Sonópolis-specific setup instructions locally.
