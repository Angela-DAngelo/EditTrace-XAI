from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SCHEMA_VERSION = "1.0"


class OperationName(StrEnum):
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    SATURATION = "saturation"
    GAUSSIAN_BLUR = "gaussian_blur"
    CROP_RESIZE = "crop_resize"
    JPEG = "jpeg"


class ImageSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    mode: Literal["RGB"] = "RGB"
    dtype: Literal["uint8"] = "uint8"
    value_min: int = 0
    value_max: int = 255


class NormalizedBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    left: float = Field(ge=0.0, le=1.0)
    top: float = Field(ge=0.0, le=1.0)
    right: float = Field(ge=0.0, le=1.0)
    bottom: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def non_empty(self) -> "NormalizedBox":
        if self.right <= self.left or self.bottom <= self.top:
            raise ValueError("crop box must have positive width and height")
        return self


class GeometryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["identity", "crop_resize"] = "identity"
    crop_box: NormalizedBox | None = None
    interpolation: Literal["nearest", "bilinear", "bicubic", "lanczos"] | None = None

    @model_validator(mode="after")
    def fields_match_kind(self) -> "GeometryRecord":
        if self.kind == "crop_resize" and self.crop_box is None:
            raise ValueError("crop_resize geometry requires crop_box")
        if self.kind == "identity" and self.crop_box is not None:
            raise ValueError("identity geometry cannot contain crop_box")
        return self


class SupportRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["global", "box"]
    box: NormalizedBox | None = None

    @model_validator(mode="after")
    def fields_match_kind(self) -> "SupportRecord":
        if self.kind == "box" and self.box is None:
            raise ValueError("box support requires box")
        if self.kind == "global" and self.box is not None:
            raise ValueError("global support cannot contain box")
        return self


class CodecRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    library: str = "Pillow"
    library_version: str
    quality: int | None = Field(default=None, ge=1, le=100)
    subsampling: int = Field(default=0, ge=0, le=2)
    optimize: bool = False
    progressive: bool = False
    bypass: bool = False


class ActionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    trace_id: str = Field(min_length=1)
    action_id: str = Field(min_length=1)
    index: int = Field(ge=0)
    op_name: OperationName
    op_version: str = "1.0"
    params: dict[str, Any]
    identity_params: dict[str, Any]
    input_spec: ImageSpec
    output_spec: ImageSpec
    geometry: GeometryRecord = Field(default_factory=GeometryRecord)
    support: SupportRecord
    codec: CodecRecord | None = None
    predecessor_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int = Field(ge=0)

    @model_validator(mode="after")
    def operation_metadata(self) -> "ActionRecord":
        if self.op_name == OperationName.CROP_RESIZE and self.geometry.kind != "crop_resize":
            raise ValueError("crop_resize action requires crop_resize geometry")
        if self.op_name != OperationName.CROP_RESIZE and self.geometry.kind != "identity":
            raise ValueError("non-geometric action requires identity geometry")
        if self.op_name == OperationName.JPEG and self.codec is None:
            raise ValueError("jpeg action requires codec metadata")
        if self.op_name != OperationName.JPEG and self.codec is not None:
            raise ValueError("codec metadata is only valid for jpeg")
        return self


class TraceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    trace_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_path: str | None = None
    ground_truth_label: int | None = Field(default=None, ge=0)
    model_id: str | None = None
    benchmark: Literal["B-Natural", "B-Single", "B-Interaction", "B-Oracle", "fixture"]
    seed: int = Field(ge=0)
    source_spec: ImageSpec
    source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    final_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    actions: list[ActionRecord] = Field(min_length=1, max_length=6)
    designated_action_id: str | None = None
    designated_pair: tuple[str, str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def trace_invariants(self) -> "TraceRecord":
        if any(action.trace_id != self.trace_id for action in self.actions):
            raise ValueError("all actions must reference the parent trace")
        if [a.index for a in self.actions] != list(range(len(self.actions))):
            raise ValueError("action indices must be contiguous and ordered")
        action_ids = [a.action_id for a in self.actions]
        if len(set(action_ids)) != len(action_ids):
            raise ValueError("action ids must be unique")
        op_names = [a.op_name for a in self.actions]
        if len(set(op_names)) != len(op_names):
            raise ValueError("v1.0 histories require unique operation types")
        if self.designated_action_id and self.designated_action_id not in action_ids:
            raise ValueError("designated action must belong to the trace")
        if self.designated_pair:
            if self.designated_pair[0] == self.designated_pair[1]:
                raise ValueError("designated pair must contain distinct actions")
            if not set(self.designated_pair).issubset(action_ids):
                raise ValueError("designated pair must belong to the trace")
        return self


class DatasetEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    path: str
    ground_truth_label: int | None = Field(default=None, ge=0)
    class_name: str | None = None
    model_id: str | None = None
    split: Literal["development", "test", "unassigned"] = "unassigned"
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def normalized_path(cls, value: str) -> str:
        return str(Path(value))


class GeneratorOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    benchmark: Literal["B-Natural", "B-Single", "B-Interaction", "B-Oracle"]
    accepted: bool
    attempts: int = Field(ge=1)
    trace: TraceRecord | None = None
    rejection_counts: dict[str, int] = Field(default_factory=dict)
    metrics: dict[str, float | int | bool] = Field(default_factory=dict)
