from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BoundingBox(SchemaModel):
    x1: int
    y1: int
    x2: int
    y2: int


class NormalizedBoundingBox(SchemaModel):
    x1: float
    y1: float
    x2: float
    y2: float


class MeasuredField(SchemaModel):
    value: Any
    provenance: str
    confidence: float


class ImageMetadata(SchemaModel):
    source_path: str
    file_name: str
    sha256: str
    width: int
    height: int


class ImageSummary(SchemaModel):
    short_caption: str
    long_description: str


class ObjectRecord(SchemaModel):
    label: str
    bbox_px: BoundingBox
    bbox_norm: NormalizedBoundingBox
    confidence: float
    attributes: dict[str, MeasuredField] = Field(default_factory=dict)


class PersonRecord(SchemaModel):
    id: str
    bbox_px: BoundingBox
    bbox_norm: NormalizedBoundingBox
    pose_angles_deg: dict[str, float] = Field(default_factory=dict)
    landmarks: dict[str, Any] = Field(default_factory=dict)
    facing_direction: MeasuredField | None = None
    gesture_hints: list[str] = Field(default_factory=list)


class FaceRecord(SchemaModel):
    id: str
    bbox_px: BoundingBox
    bbox_norm: NormalizedBoundingBox
    landmarks: dict[str, Any] = Field(default_factory=dict)
    blendshapes: dict[str, Any] = Field(default_factory=dict)
    inferred_expression: MeasuredField | None = None


class TextRegion(SchemaModel):
    text: str
    bbox_px: BoundingBox
    bbox_norm: NormalizedBoundingBox
    confidence: float
    provenance: str


class RegionRecord(SchemaModel):
    id: str
    label: str
    bbox_px: BoundingBox
    bbox_norm: NormalizedBoundingBox
    dominant_colors: list[MeasuredField] = Field(default_factory=list)


class SceneSummary(SchemaModel):
    category: str
    confidence: float
    provenance: str


class UncertaintyRecord(SchemaModel):
    source: str
    message: str


class DescriptionLayer(SchemaModel):
    key: str
    title: str
    text: str
    file_name: str


class AnalysisEvent(SchemaModel):
    timestamp: datetime
    stage: str
    status: str
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)


class AnalysisResult(SchemaModel):
    image: ImageMetadata
    summary: ImageSummary
    objects: list[ObjectRecord] = Field(default_factory=list)
    people: list[PersonRecord] = Field(default_factory=list)
    faces: list[FaceRecord] = Field(default_factory=list)
    regions: list[RegionRecord] = Field(default_factory=list)
    text_regions: list[TextRegion] = Field(default_factory=list)
    scene: SceneSummary
    provenance: dict[str, list[str]] = Field(default_factory=dict)
    uncertainty: list[UncertaintyRecord] = Field(default_factory=list)
    description_layers: list[DescriptionLayer] = Field(default_factory=list)
    orchestration_events: list[AnalysisEvent] = Field(default_factory=list)
