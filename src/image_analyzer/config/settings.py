from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PathConfig:
    artifact_dir: Path
    log_dir: Path


@dataclass(frozen=True)
class PipelineConfig:
    save_debug_by_default: bool
    dense_caption_first: bool
    enable_ocr: bool
    enable_pose: bool
    enable_face: bool
    enable_segmentation: bool


@dataclass(frozen=True)
class ModelConfig:
    ollama_base_url: str
    synthesis_model: str
    ollama_timeout_sec: int
    ollama_max_loaded_models: int
    ollama_num_parallel: int


@dataclass(frozen=True)
class ProviderConfig:
    detector: str
    region_captioner: str
    pose: str
    face: str
    segmentation: str
    ocr: str
    color: str


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    paths: PathConfig
    pipeline: PipelineConfig
    models: ModelConfig
    providers: ProviderConfig


def load_settings(project_root: Path) -> AppConfig:
    payload = _read_yaml(project_root / "configs" / "settings.yaml")
    return AppConfig(
        project_root=project_root,
        paths=PathConfig(
            artifact_dir=project_root / payload["paths"]["artifact_dir"],
            log_dir=project_root / payload["paths"]["log_dir"],
        ),
        pipeline=PipelineConfig(
            save_debug_by_default=bool(payload["pipeline"]["save_debug_by_default"]),
            dense_caption_first=bool(payload["pipeline"]["dense_caption_first"]),
            enable_ocr=bool(payload["pipeline"]["enable_ocr"]),
            enable_pose=bool(payload["pipeline"]["enable_pose"]),
            enable_face=bool(payload["pipeline"]["enable_face"]),
            enable_segmentation=bool(payload["pipeline"]["enable_segmentation"]),
        ),
        models=ModelConfig(
            ollama_base_url=str(payload["models"]["ollama_base_url"]),
            synthesis_model=str(payload["models"]["synthesis_model"]),
            ollama_timeout_sec=int(payload["models"]["ollama_timeout_sec"]),
            ollama_max_loaded_models=int(payload["models"]["ollama_max_loaded_models"]),
            ollama_num_parallel=int(payload["models"]["ollama_num_parallel"]),
        ),
        providers=ProviderConfig(
            detector=str(payload["providers"]["detector"]),
            region_captioner=str(payload["providers"]["region_captioner"]),
            pose=str(payload["providers"]["pose"]),
            face=str(payload["providers"]["face"]),
            segmentation=str(payload["providers"]["segmentation"]),
            ocr=str(payload["providers"]["ocr"]),
            color=str(payload["providers"]["color"]),
        ),
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))

