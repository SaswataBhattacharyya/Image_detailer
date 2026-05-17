# Pipeline

1. Load image metadata.
2. Run detector and region captioner.
3. Route person-related modules only if a person is present.
4. Run OCR and color extraction.
5. Synthesize a final detailed description from measured findings.
6. Save `details.json`, `description.txt`, and optional `module_outputs.json`.

