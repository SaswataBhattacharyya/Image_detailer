# Knowledge

The repo combines specialist modules:

- object detection
- multi-pass VLM image analysis
- OCR
- color extraction
- optional pose / face / segmentation
- prompt package generation
- optional local image generation
- VLM image comparison
- hybrid similarity scoring

Important artifact classes:

- scene map JSON
- visual hierarchy JSON
- prompt package JSON/text
- generation metadata
- comparison report JSON
- hybrid score JSON
- final run report

The final outputs should preserve which module or phase produced each field and
how confident it was.
