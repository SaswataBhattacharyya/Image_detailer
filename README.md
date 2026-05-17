# image_analyzer

Local-first image analysis pipeline that produces:

- measured-first JSON
- detailed text descriptions
- layered text descriptions
- orchestration event logs
- optional per-module debug output

This directory is intended to work as its own project root and its own git repo.
It does not import from the surrounding `vid_analyzer` project.

## What this repo does

The pipeline does not rely on one captioning model alone. It coordinates multiple specialists:

- object and region discovery
- optional pose / face / segmentation analysis
- OCR
- dominant color extraction
- local Ollama synthesis for final narrative text
- orchestration events that explain which modules ran and why

## Quick start

```bash
chmod +x scripts/*.sh
./scripts/bootstrap_vm.sh
./scripts/run_streamlit_app.sh
```

This repo is intended to be operated with only two scripts:

- `./scripts/bootstrap_vm.sh`
  Sets up the full VM environment end to end. It installs system packages, creates the Python virtual environment, installs Python dependencies, installs Ollama, pulls the default models, installs OpenClaw, and ends with the OpenClaw setup checks.
- `./scripts/run_streamlit_app.sh`
  Starts the Streamlit UI for the app.

For the normal VM workflow, these are the only two scripts you need.

## Why there are other `.sh` files

Some older helper scripts are still in the repo because the project was originally kept modular:

- `./scripts/run_image_analyzer.sh`
  Runs the analyzer directly from the terminal without Streamlit. This is optional and is only useful for CLI-only testing or debugging.
- `./scripts/setup_openclaw_supervisor.sh`
  Performs standalone setup checks for the separate OpenClaw supervisor path. This is not required for the normal Streamlit workflow because `bootstrap_vm.sh` already handles the main environment setup.
- `./scripts/run_openclaw_supervisor.sh`
  Starts a small FastAPI supervisor service used by the older supervisor path. This is not required for normal usage of the app.

So in practice:

- required for normal use: `bootstrap_vm.sh`, `run_streamlit_app.sh`
- optional helper scripts: everything else in `scripts/`

Defaults:

- UI host: `0.0.0.0`
- UI port: `8501`
- override with `STREAMLIT_HOST` and `STREAMLIT_PORT`

If you want a separate repository boundary here:

```bash
git init
```

## Entry points

- `./scripts/run_streamlit_app.sh`
- `./scripts/bootstrap_vm.sh`

Optional helper entry points:

- `./scripts/run_image_analyzer.sh analyze-image <image_path>`
- `./scripts/run_image_analyzer.sh analyze-batch <input_dir>`
- `./scripts/setup_openclaw_supervisor.sh`
- `./scripts/run_openclaw_supervisor.sh`

## Output bundle

Each analyzed image writes a folder under `artifacts/` with:

- `details.json`
- `description.txt`
- `events.json`
- `layers.json`
- `<image_stem>_desc_1.txt` ... `<image_stem>_desc_6.txt`
- `module_outputs.json`

The layered files are fixed in this order:

1. `summary`
2. `detailed_description`
3. `colors_and_materials`
4. `composition_and_camera`
5. `emotion_style_or_intent`
6. `ocr_and_context`

## Streamlit UI

The Streamlit app supports:

- single-image analysis by local path
- single-image upload
- batch analysis by folder or manifest path
- live stage/event updates while analysis runs
- per-image layered descriptions, structured JSON, and orchestration events

For batch runs, the UI shows the current image, the latest event, a rolling event log, and the latest completed result bundle.

## Setup policy

- Linux + NVIDIA GPU is the primary target.
- The bootstrap script validates GPU visibility and driver health.
- Host GPU drivers are validated but not auto-installed.
- Ollama is used for local synthesis and is configured to keep only one model loaded at a time by default.

## Standalone layout

- `src/image_analyzer/` core package
- `configs/` local settings
- `scripts/` setup and launch wrappers
- `openclaw/` repo-local OpenClaw instructions
- `tests/` standalone verification
- `pyproject.toml` package metadata for editable install from this root

## VM workflow

```bash
./scripts/bootstrap_vm.sh
./scripts/run_streamlit_app.sh
```

Then open `http://<vm-ip>:8501`.

If the UI produces fallback text instead of model-rich output, check:

- `ollama` is reachable on `http://127.0.0.1:11434`
- required models were pulled by `bootstrap_vm.sh`
- provider/model downloads are available when expected
- host GPU drivers are visible if you expect GPU execution
