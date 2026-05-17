from __future__ import annotations

import os
from pathlib import Path

import requests

from image_analyzer.providers.base import OptionalProvider, ProviderArtifact


class OllamaSynthesisProvider(OptionalProvider):
    name = "ollama"

    def __init__(self, base_url: str, model_name: str, timeout_sec: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_sec = timeout_sec

    def synthesize(self, image_path: Path, context: dict[str, object]) -> ProviderArtifact:
        payload = {
            "model": self.model_name,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": _build_prompt(context),
                }
            ],
            "options": {"num_predict": 700},
            "keep_alive": "15m",
        }
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout_sec,
            )
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "").strip()
            if content:
                return ProviderArtifact(provider=self.name, data={"description": content})
        except Exception as exc:
            return ProviderArtifact(provider=self.name, warnings=[f"Ollama synthesis unavailable: {exc}"])
        return ProviderArtifact(provider=self.name, warnings=["Ollama synthesis returned no content"])


def configure_ollama_runtime(max_loaded_models: int, num_parallel: int) -> None:
    os.environ.setdefault("OLLAMA_MAX_LOADED_MODELS", str(max_loaded_models))
    os.environ.setdefault("OLLAMA_NUM_PARALLEL", str(num_parallel))


def _build_prompt(context: dict[str, object]) -> str:
    return (
        "Write a precise, richly detailed image description from the structured analysis below. "
        "Prefer measured findings. Call out uncertainty explicitly. "
        f"Structured analysis: {context}"
    )

