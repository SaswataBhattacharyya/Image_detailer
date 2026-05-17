from __future__ import annotations

import os
from pathlib import Path

from image_analyzer.models.schemas import BoundingBox
from image_analyzer.providers.base import OptionalProvider, ProviderArtifact


class YoloDetectionProvider(OptionalProvider):
    name = "yolo"

    def analyze(self, image_path: Path, context: dict[str, object]) -> ProviderArtifact:
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:
            return ProviderArtifact(provider=self.name, warnings=[f"YOLO unavailable: {exc}"])

        model_path = os.environ.get("IMAGE_ANALYZER_YOLO_MODEL", "yolov8n.pt")
        allow_downloads = os.environ.get("IMAGE_ANALYZER_ALLOW_MODEL_DOWNLOADS", "").lower() in {"1", "true", "yes"}
        if not allow_downloads and not Path(model_path).exists():
            return ProviderArtifact(
                provider=self.name,
                warnings=[f"YOLO model not found locally: {model_path}. Set IMAGE_ANALYZER_ALLOW_MODEL_DOWNLOADS=1 to permit auto-downloads."],
            )

        try:
            model = YOLO(model_path)
            results = model.predict(source=str(image_path), verbose=False)
            detections = []
            for result in results:
                boxes = getattr(result, "boxes", None)
                if boxes is None:
                    continue
                for box in boxes:
                    coords = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0].item())
                    label = result.names.get(cls_id, str(cls_id))
                    detections.append(
                        {
                            "label": label,
                            "bbox_px": BoundingBox(
                                x1=int(coords[0]),
                                y1=int(coords[1]),
                                x2=int(coords[2]),
                                y2=int(coords[3]),
                            ).model_dump(),
                            "confidence": float(box.conf[0].item()),
                            "provenance": self.name,
                        }
                    )
            return ProviderArtifact(provider=self.name, data={"detections": detections})
        except Exception as exc:
            return ProviderArtifact(provider=self.name, warnings=[f"YOLO failed: {exc}"])


class FlorenceRegionProvider(OptionalProvider):
    name = "florence2"

    def analyze(self, image_path: Path, context: dict[str, object]) -> ProviderArtifact:
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore
            import torch  # type: ignore
            from PIL import Image
        except Exception as exc:
            return ProviderArtifact(provider=self.name, warnings=[f"Florence-2 unavailable: {exc}"])

        allow_downloads = os.environ.get("IMAGE_ANALYZER_ALLOW_MODEL_DOWNLOADS", "").lower() in {"1", "true", "yes"}
        try:
            processor = AutoProcessor.from_pretrained(
                "microsoft/Florence-2-large",
                trust_remote_code=True,
                local_files_only=not allow_downloads,
            )
            model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Florence-2-large",
                trust_remote_code=True,
                local_files_only=not allow_downloads,
            )
            image = Image.open(image_path).convert("RGB")
            prompt = "<MORE_DETAILED_CAPTION>"
            inputs = processor(text=prompt, images=image, return_tensors="pt")
            generated = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=128,
                num_beams=2,
            )
            text = processor.batch_decode(generated, skip_special_tokens=False)[0]
            parsed = processor.post_process_generation(
                text,
                task=prompt,
                image_size=(image.width, image.height),
            )
            return ProviderArtifact(provider=self.name, data={"caption": parsed})
        except Exception as exc:
            return ProviderArtifact(provider=self.name, warnings=[f"Florence-2 failed: {exc}"])


class TesseractOcrProvider(OptionalProvider):
    name = "tesseract"

    def analyze(self, image_path: Path, context: dict[str, object]) -> ProviderArtifact:
        try:
            import pytesseract  # type: ignore
            from PIL import Image
        except Exception as exc:
            return ProviderArtifact(provider=self.name, warnings=[f"Tesseract unavailable: {exc}"])

        try:
            with Image.open(image_path) as image:
                text = pytesseract.image_to_string(image).strip()
            return ProviderArtifact(provider=self.name, data={"ocr_text": text})
        except Exception as exc:
            return ProviderArtifact(provider=self.name, warnings=[f"Tesseract failed: {exc}"])
