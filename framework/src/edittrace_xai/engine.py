from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping

from PIL import Image

from .operations import OperationValidationError, apply_operation, attenuate_params, ensure_rgb_uint8
from .schema import ActionRecord, ImageSpec, TraceRecord


class ReplayError(RuntimeError):
    def __init__(self, code: str, message: str, action_id: str | None = None):
        super().__init__(message)
        self.code = code
        self.action_id = action_id


def image_spec(image: Image.Image) -> ImageSpec:
    image = ensure_rgb_uint8(image)
    return ImageSpec(width=image.width, height=image.height)


def image_hash(image: Image.Image) -> str:
    image = ensure_rgb_uint8(image)
    digest = hashlib.sha256()
    digest.update(image.mode.encode("ascii"))
    digest.update(image.width.to_bytes(8, "big"))
    digest.update(image.height.to_bytes(8, "big"))
    digest.update(image.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ReplayResult:
    image: Image.Image
    output_hash: str
    executed_action_ids: tuple[str, ...]
    removed_action_ids: tuple[str, ...]
    override_action_ids: tuple[str, ...]


class ReplayEngine:
    def replay(
        self,
        source: Image.Image,
        trace: TraceRecord,
        *,
        remove_action_ids: set[str] | None = None,
        parameter_overrides: Mapping[str, Mapping[str, object]] | None = None,
        verify_original_lineage: bool = False,
    ) -> ReplayResult:
        source = ensure_rgb_uint8(source)
        removals = set(remove_action_ids or set())
        overrides = dict(parameter_overrides or {})
        known = {action.action_id for action in trace.actions}
        unknown = (removals | set(overrides)) - known
        if unknown:
            raise ReplayError("unknown_action", f"unknown actions: {sorted(unknown)}")
        if removals & set(overrides):
            raise ReplayError("conflicting_intervention", "an action cannot be removed and overridden")
        if image_spec(source) != trace.source_spec:
            raise ReplayError("source_spec_mismatch", "source image does not match trace source_spec")
        if image_hash(source) != trace.source_hash:
            raise ReplayError("source_hash_mismatch", "source image does not match trace source_hash")

        current = source.copy()
        executed: list[str] = []
        for action in trace.actions:
            if action.action_id in removals:
                continue
            if image_spec(current) != action.input_spec:
                raise ReplayError(
                    "input_spec_mismatch",
                    f"replay input does not match action input_spec: {action.action_id}",
                    action.action_id,
                )
            if verify_original_lineage and not removals and not overrides:
                if image_hash(current) != action.predecessor_hash:
                    raise ReplayError(
                        "lineage_mismatch",
                        f"predecessor hash mismatch: {action.action_id}",
                        action.action_id,
                    )
            params = overrides.get(action.action_id, action.params)
            try:
                current = apply_operation(current, action.op_name, params).image
            except (KeyError, TypeError, ValueError, OperationValidationError) as exc:
                raise ReplayError("invalid_operation", str(exc), action.action_id) from exc
            if image_spec(current) != action.output_spec:
                raise ReplayError(
                    "output_spec_mismatch",
                    f"replay output does not match action output_spec: {action.action_id}",
                    action.action_id,
                )
            if verify_original_lineage and not removals and not overrides:
                if image_hash(current) != action.output_hash:
                    raise ReplayError(
                        "output_hash_mismatch",
                        f"output hash mismatch: {action.action_id}",
                        action.action_id,
                    )
            executed.append(action.action_id)

        return ReplayResult(
            image=current,
            output_hash=image_hash(current),
            executed_action_ids=tuple(executed),
            removed_action_ids=tuple(sorted(removals)),
            override_action_ids=tuple(sorted(overrides)),
        )

    def remove_action(self, source: Image.Image, trace: TraceRecord, action_id: str) -> ReplayResult:
        return self.replay(source, trace, remove_action_ids={action_id})

    def remove_pair(
        self,
        source: Image.Image,
        trace: TraceRecord,
        action_i: str,
        action_j: str,
    ) -> ReplayResult:
        if action_i == action_j:
            raise ReplayError("invalid_pair", "pair members must be distinct")
        return self.replay(source, trace, remove_action_ids={action_i, action_j})

    def attenuate_action(
        self,
        source: Image.Image,
        trace: TraceRecord,
        action_id: str,
        alpha: float,
    ) -> ReplayResult:
        action = next((item for item in trace.actions if item.action_id == action_id), None)
        if action is None:
            raise ReplayError("unknown_action", f"unknown action: {action_id}")
        params = attenuate_params(action.op_name, action.params, alpha)
        return self.replay(source, trace, parameter_overrides={action_id: params})

