# Pipeline

1. Load image metadata and create a timestamped run folder.
2. Run multi-pass VLM analysis on the reference image.
3. Build a structured scene map and refined scene map.
4. Build a visual hierarchy and prompt package.
5. If generation is disabled, stop after writing prompt/report artifacts.
6. If generation is enabled, call the repo generator wrapper:
   `scripts/run_qwen_image_generation.sh`
7. Compare generated image against reference using:
   - VLM semantic comparison
   - code-based perceptual scoring
8. Compute a hybrid similarity score.
9. If the threshold is met, stop and mark the best iteration.
10. Otherwise generate prompt corrections and repeat until threshold or max iterations.
11. Save run report, final prompt package, iteration artifacts, and logs.
