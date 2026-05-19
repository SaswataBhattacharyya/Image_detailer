from __future__ import annotations

import base64
import os
import json
from pathlib import Path
from typing import Any

import requests

from image_analyzer.providers.base import OptionalProvider, ProviderArtifact


class OllamaSynthesisProvider(OptionalProvider):
    name = "ollama"

    def __init__(self, base_url: str, model_name: str, timeout_sec: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_sec = timeout_sec

    def synthesize(self, image_path: Path, context: dict[str, object]) -> ProviderArtifact:
        try:
            content = self._chat(
                messages=[
                    {
                        "role": "user",
                        "content": _build_prompt(context),
                    }
                ],
                image_path=None,
                num_predict=700,
            )
            if content:
                return ProviderArtifact(provider=self.name, data={"description": content})
        except Exception as exc:
            return ProviderArtifact(provider=self.name, warnings=[f"Ollama synthesis unavailable: {exc}"])
        return ProviderArtifact(provider=self.name, warnings=["Ollama synthesis returned no content"])

    def analyze_visual_pass(self, image_path: Path, prompt: str, *, num_predict: int = 900) -> ProviderArtifact:
        try:
            content = self._chat(
                messages=[{"role": "user", "content": prompt}],
                image_path=image_path,
                num_predict=num_predict,
            )
            return ProviderArtifact(provider=self.name, data={"content": content})
        except Exception as exc:
            return ProviderArtifact(provider=self.name, warnings=[f"Ollama visual analysis unavailable: {exc}"])

    def generate_json(self, prompt: str, *, image_path: Path | None = None, num_predict: int = 1200) -> ProviderArtifact:
        try:
            content = self._chat(
                messages=[{"role": "user", "content": prompt}],
                image_path=image_path,
                num_predict=num_predict,
                format_mode="json",
            )
            parsed = _safe_parse_json(content)
            if parsed is not None:
                return ProviderArtifact(provider=self.name, data={"json": parsed, "content": content})
            return ProviderArtifact(
                provider=self.name,
                data={"content": content},
                warnings=["Ollama JSON response could not be parsed cleanly."],
            )
        except Exception as exc:
            return ProviderArtifact(provider=self.name, warnings=[f"Ollama JSON generation unavailable: {exc}"])

    def compare_images(self, reference_path: Path, generated_path: Path, prompt: str) -> ProviderArtifact:
        content_prompt = (
            f"{prompt}\n\nReference image is attached first. Generated image is attached second. "
            "Produce strict JSON if possible."
        )
        try:
            content = self._chat(
                messages=[
                    {
                        "role": "user",
                        "content": content_prompt,
                        "images": [_encode_image(reference_path), _encode_image(generated_path)],
                    }
                ],
                image_path=None,
                num_predict=1200,
                format_mode="json",
            )
            parsed = _safe_parse_json(content)
            if parsed is not None:
                return ProviderArtifact(provider=self.name, data={"json": parsed, "content": content})
            return ProviderArtifact(provider=self.name, data={"content": content}, warnings=["Comparison JSON was not parseable."])
        except Exception as exc:
            return ProviderArtifact(provider=self.name, warnings=[f"Ollama image comparison unavailable: {exc}"])

    def _chat(
        self,
        *,
        messages: list[dict[str, Any]],
        image_path: Path | None,
        num_predict: int,
        format_mode: str | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "stream": False,
            "messages": [dict(message) for message in messages],
            "options": {"num_predict": num_predict},
            "keep_alive": "15m",
        }
        if format_mode == "json":
            payload["format"] = "json"
        if image_path is not None and payload["messages"]:
            payload["messages"][0]["images"] = [_encode_image(image_path)]
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout_sec,
        )
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "").strip()


def configure_ollama_runtime(max_loaded_models: int, num_parallel: int) -> None:
    os.environ.setdefault("OLLAMA_MAX_LOADED_MODELS", str(max_loaded_models))
    os.environ.setdefault("OLLAMA_NUM_PARALLEL", str(num_parallel))


def _build_prompt(context: dict[str, object]) -> str:
    return (
        "Write a precise, richly detailed image description from the structured analysis below. "
        "Prefer measured findings. Call out uncertainty explicitly. "
        f"Structured analysis: {context}"
    )


def _encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _safe_parse_json(content: str) -> dict[str, Any] | list[Any] | None:
    if not content.strip():
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = min(
            (index for index in (content.find("{"), content.find("[")) if index != -1),
            default=-1,
        )
        end = max(content.rfind("}"), content.rfind("]"))
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            return None
