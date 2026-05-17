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
chmod +x scripts/bootstrap_vm.sh scripts/run_image_analyzer.sh scripts/run_streamlit_app.sh scripts/setup_openclaw_supervisor.sh scripts/run_openclaw_supervisor.sh
./scripts/bootstrap_vm.sh
./scripts/run_image_analyzer.sh analyze-image path/to/image.jpg
```

To run the Streamlit UI on a VM:

```bash
./scripts/run_streamlit_app.sh
```

Defaults:

- UI host: `0.0.0.0`
- UI port: `8501`
- override with `STREAMLIT_HOST` and `STREAMLIT_PORT`

If you want a separate repository boundary here:

```bash
git init
```

## Entry points

- `./scripts/run_image_analyzer.sh analyze-image <image_path>`
- `./scripts/run_image_analyzer.sh analyze-batch <input_dir>`
- `./scripts/run_streamlit_app.sh`

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
