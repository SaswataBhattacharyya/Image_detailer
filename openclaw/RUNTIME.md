# Runtime

Runtime defaults:

- Ollama at `http://127.0.0.1:11434`
- VLM: `qwen2.5vl:7b`
- structuring model: `qwen2.5-coder:14b`
- reasoning/default supervisor model: `qwen2.5-coder:14b`
- one loaded model at a time
- Linux + NVIDIA GPU is the primary target

Generation runtime:

- wrapper: `./scripts/run_qwen_image_generation.sh`
- backend env var: `IMAGE_ANALYZER_QWEN_IMAGE_RUNNER`

If the generation backend is missing, fail clearly.
If the GPU is not visible, stop and report the host prerequisite gap.
If Ollama returns a model-load/resource-limitation error, unload other active Ollama models and retry once before surfacing the failure.
