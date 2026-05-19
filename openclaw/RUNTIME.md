# Runtime

Runtime defaults:

- Ollama on `127.0.0.1:11434`
- one loaded model at a time
- one parallel request by default
- Linux + NVIDIA GPU primary target
- advanced VLM model: `qwen2.5vl:32b`
- structuring model: `qwen2.5-coder:14b`

If the GPU is not visible, stop and report the missing host-level prerequisites instead of silently falling back.

Generation runtime notes:

- The closed-loop generator path is wired through:
  `scripts/run_qwen_image_generation.sh`
- The repo expects either:
  - `IMAGE_ANALYZER_QWEN_IMAGE_RUNNER`
  - or a repo-local replacement implementation behind that wrapper
- The generator wrapper receives prompt path, negative prompt path, output image
  path, size, steps, cfg, sampler, seed, and model.
- If generation is requested but no backend command is configured, the run should
  fail clearly as configuration incomplete, not silently succeed.
