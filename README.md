# image_analyzer

Local-first image analysis and closed-loop image recreation pipeline.

This repo now supports two distinct modes:

- `analyze-image`: legacy measured-first image analysis with JSON/text/layered outputs
- `analyze-detailed`: advanced multi-pass VLM analysis that can run a closed loop:
  `analyze -> scene map -> prompt package -> generate -> compare -> correct -> repeat`

## What the repo does now

The advanced path is built around:

- multi-pass image-grounded VLM analysis
- structured scene-map JSON generation
- visual hierarchy and prompt package generation
- optional local image generation through a stable wrapper script
- hybrid similarity scoring:
  - semantic/VLM comparison
  - code-based perceptual scoring
- iterative prompt correction until threshold or max iterations

The simple path still exists for lightweight image analysis and Streamlit usage.

## Quick start

```bash
chmod +x scripts/*.sh
./scripts/bootstrap_vm.sh
./scripts/run_streamlit_app.sh
```

The Streamlit UI is still useful for the legacy/simple analyzer path.

For the advanced closed-loop path, use the CLI:

```bash
image-analyzer analyze-detailed /path/to/image.png
```

## Current setup behavior

`./scripts/bootstrap_vm.sh` does the following:

- installs base system packages
- creates the Python virtual environment
- installs the repo in editable mode
- installs Ollama if missing
- ensures Ollama is reachable
- pulls default Ollama models:
  - `qwen2.5vl:32b`
  - `qwen2.5-coder:14b`
  - `gemma4:latest`
- installs OpenClaw if missing
- runs the Qwen-image backend setup helper

Important:

- the **closed-loop engine is implemented**
- but actual image generation still depends on a local backend runner command
- `bootstrap_vm.sh` will warn you if that backend is not configured

## Qwen-image backend setup

The repo uses this stable wrapper for generation:

```bash
./scripts/run_qwen_image_generation.sh
```

The closed-loop pipeline calls that wrapper by default through the config.

You must provide the real backend in one of these ways:

1. Configure a setup/install command before bootstrap:

```bash
export IMAGE_ANALYZER_QWEN_IMAGE_SETUP_COMMAND='...your install command...'
./scripts/bootstrap_vm.sh
```

2. Configure the runtime generation command:

```bash
export IMAGE_ANALYZER_QWEN_IMAGE_RUNNER='...your qwen-image generation command...'
```

When invoked, the wrapper exposes these environment variables to your command:

- `IMAGE_ANALYZER_QWEN_IMAGE_PROMPT_FILE`
- `IMAGE_ANALYZER_QWEN_IMAGE_NEGATIVE_PROMPT_FILE`
- `IMAGE_ANALYZER_QWEN_IMAGE_OUTPUT_IMAGE`
- `IMAGE_ANALYZER_QWEN_IMAGE_WIDTH`
- `IMAGE_ANALYZER_QWEN_IMAGE_HEIGHT`
- `IMAGE_ANALYZER_QWEN_IMAGE_STEPS`
- `IMAGE_ANALYZER_QWEN_IMAGE_CFG`
- `IMAGE_ANALYZER_QWEN_IMAGE_SAMPLER`
- `IMAGE_ANALYZER_QWEN_IMAGE_SEED`
- `IMAGE_ANALYZER_QWEN_IMAGE_MODEL`

If you prefer, you can replace the wrapper with a repo-local direct implementation.

## Advanced CLI

Basic detailed run:

```bash
image-analyzer analyze-detailed /path/to/image.png
```

Closed-loop run:

```bash
image-analyzer analyze-detailed /path/to/image.png \
  --enable-generation \
  --enable-comparison \
  --target-score 0.95 \
  --max-iterations 5
```

Useful flags:

- `--output-dir`
- `--project-name`
- `--iterations`
- `--aspect-ratio`
- `--enable-generation`
- `--enable-comparison`
- `--target-score`
- `--max-iterations`

## Output layout

Legacy/simple runs still write under `artifacts/<image_stem>/`.

Detailed closed-loop runs write timestamped run folders under:

```text
artifacts/detailed_runs/<timestamp>_<project>_<image>/
```

Each run folder contains:

```text
input/
analysis/
prompts/
generated/
comparisons/
reports/
logs/
```

Typical advanced outputs include:

- `analysis/02_structured_scene_map.json`
- `analysis/03_refined_scene_map.json`
- `analysis/04_visual_hierarchy.json`
- `prompts/final_prompt.txt`
- `reports/final_prompt_package.json`
- `reports/run_report.json`
- `logs/model_calls.jsonl`

When generation/comparison are enabled, you also get iteration artifacts like:

- `generated/generated_v1.png`
- `comparisons/comparison_v1.json`
- `comparisons/correction_v1.json`
- `comparisons/hybrid_score_v1.json`

## Streamlit

The Streamlit app still reflects the simple analyzer path.

It is **not** yet the primary interface for:

- closed-loop generation
- iteration control
- hybrid similarity scoring
- prompt correction rounds

Use the CLI for the advanced workflow.

## OpenClaw

Repo-local OpenClaw instructions live under `openclaw/`.

OpenClaw should treat this repo as the main execution engine:

- OpenClaw can start detailed runs
- OpenClaw can inspect run artifacts
- OpenClaw should not own the closed-loop logic itself

## Setup policy

- Linux + NVIDIA GPU is the primary target
- one heavy model on GPU at a time is the intended runtime policy
- host GPU drivers are validated but not auto-installed
- Ollama is configured for one loaded model at a time by default

## Entry points

- `./scripts/bootstrap_vm.sh`
- `./scripts/run_streamlit_app.sh`
- `./scripts/run_image_analyzer.sh analyze-image <image_path>`
- `image-analyzer analyze-detailed <image_path>`
- `./scripts/setup_openclaw_supervisor.sh`
- `./scripts/run_openclaw_supervisor.sh`

## If advanced runs fail

Check:

- `ollama` is reachable at `http://127.0.0.1:11434`
- `qwen2.5vl:32b` is pulled and fits on your GPU
- the generator backend is configured
- `IMAGE_ANALYZER_QWEN_IMAGE_RUNNER` is set if generation is enabled
- your runner actually writes the expected output image
- the VM firewall/security group exposes the Streamlit port if you are using the UI
