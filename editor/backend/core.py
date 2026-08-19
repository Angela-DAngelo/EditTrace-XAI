from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


EDIT_SPECS: dict[str, dict[str, Any]] = {
    "brightness": {"label": "Brightness", "param": "factor", "identity": 1.0, "min": 0.25, "max": 1.75, "step": 0.05},
    "contrast": {"label": "Contrast", "param": "factor", "identity": 1.0, "min": 0.25, "max": 1.75, "step": 0.05},
    "saturation": {"label": "Saturation", "param": "factor", "identity": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
    "blur": {"label": "Gaussian blur", "param": "radius", "identity": 0.0, "min": 0.0, "max": 12.0, "step": 0.25},
    "crop": {"label": "Crop + resize", "identity": None},
    "jpeg": {"label": "JPEG compression", "param": "quality", "identity": None, "min": 5, "max": 100, "step": 1},
}


@dataclass
class Action:
    id: str
    type: str
    params: dict[str, Any]
    input_size: tuple[int, int]
    output_size: tuple[int, int]
    expected_output_hash: str | None = None
    computed_output_hash: str | None = None
    hash_match: bool | None = None
    seed: int | None = None
    op_version: str = "1.0"


@dataclass
class Snapshot:
    actions: list[Action]


@dataclass
class Session:
    id: str
    source: Image.Image
    filename: str
    target_class: str
    ground_truth_label: int | None = None
    trace_id: str | None = None
    benchmark: str | None = None
    source_id: str | None = None
    expected_source_hash: str | None = None
    computed_source_hash: str | None = None
    source_hash_match: bool | None = None
    eligible: bool | None = None
    eligibility_reason: str | None = None
    replay_valid: bool | None = None
    replay_reason: str | None = None
    frozen_hashes_matching: int = 0
    frozen_hashes_checked: int = 0
    dataset_mode: bool = False
    actions: list[Action] = field(default_factory=list)
    undo: list[Snapshot] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


class SurrogateModel:
    """Deterministic visual-feature model used only to exercise the workflow."""

    labels = ("warm", "cool", "bright", "dark", "vivid", "muted", "detailed", "soft")

    def logits(self, image: Image.Image) -> dict[str, float]:
        arr = np.asarray(image.resize((128, 128)), dtype=np.float32) / 255.0
        mean = arr.mean(axis=(0, 1))
        luminance = float((arr * np.array([0.2126, 0.7152, 0.0722])).sum(axis=2).mean())
        contrast = float(arr.std())
        saturation = float((arr.max(axis=2) - arr.min(axis=2)).mean())
        gx = np.abs(np.diff(arr, axis=1)).mean()
        gy = np.abs(np.diff(arr, axis=0)).mean()
        detail = float((gx + gy) * 4.5)
        warmth = float(mean[0] - mean[2])
        scores = {
            "warm": 2.8 * warmth + 0.4 * saturation,
            "cool": -2.8 * warmth + 0.4 * saturation,
            "bright": 2.5 * (luminance - 0.45) + 0.2 * contrast,
            "dark": 2.5 * (0.55 - luminance) + 0.2 * contrast,
            "vivid": 3.0 * saturation + 0.2 * contrast,
            "muted": 1.4 - 3.0 * saturation,
            "detailed": 2.4 * detail + 0.4 * contrast,
            "soft": 1.25 - 2.4 * detail,
        }
        return {k: round(v, 6) for k, v in scores.items()}

    def prediction(self, image: Image.Image, target: str | None = None) -> dict[str, Any]:
        logits = self.logits(image)
        ordered = sorted(logits.items(), key=lambda item: (-item[1], item[0]))
        predicted, predicted_logit = ordered[0]
        target = target or predicted
        competitor, competitor_logit = next(item for item in ordered if item[0] != target)
        values = np.array(list(logits.values()), dtype=np.float64)
        probs = np.exp(values - values.max())
        probs /= probs.sum()
        probability = dict(zip(logits, probs, strict=True))[target]
        return {
            "predicted_index": None,
            "target_index": None,
            "predicted_class": predicted,
            "target_class": target,
            "target_probability": round(float(probability), 5),
            "target_logit": logits[target],
            "competitor_class": competitor,
            "competitor_logit": competitor_logit,
            "margin": round(float(logits[target] - competitor_logit), 6),
            "recovered": bool(logits[target] - competitor_logit > 0),
        }


class TorchvisionModel:
    """Optional ImageNet adapter using the exact weight/preprocess identifiers below."""

    def __init__(self, model_id: str):
        try:
            import torch
            from torchvision.models import ResNet50_Weights, ViT_B_16_Weights, resnet50, vit_b_16
        except ImportError as exc:
            raise RuntimeError("EDITTRACE_MODEL requires torch and torchvision. Use 'surrogate' for the dependency-free demo.") from exc
        if model_id == "resnet50":
            self.weights_id = "ResNet50_Weights.IMAGENET1K_V2"
            self.weights = ResNet50_Weights.IMAGENET1K_V2
            self.network = resnet50(weights=self.weights)
        elif model_id == "vit_b_16":
            self.weights_id = "ViT_B_16_Weights.IMAGENET1K_V1"
            self.weights = ViT_B_16_Weights.IMAGENET1K_V1
            self.network = vit_b_16(weights=self.weights)
        else:
            raise ValueError("EDITTRACE_MODEL must be surrogate, resnet50, or vit_b_16.")
        self.model_id = model_id
        self.experimental = True
        self.torch = torch
        self.device = torch.device(os.getenv("EDITTRACE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
        self.network.eval().to(self.device)
        self.preprocess = self.weights.transforms()
        self.categories = self.weights.meta["categories"]

    def label_for_index(self, index: int) -> str:
        if not 0 <= index < len(self.categories):
            raise ValueError(f"ImageNet ground-truth label {index} is outside [0, 999].")
        return f"{index}: {self.categories[index]}"

    def prediction(self, image: Image.Image, target: str | None = None) -> dict[str, Any]:
        with self.torch.inference_mode():
            logits_tensor = self.network(self.preprocess(image).unsqueeze(0).to(self.device))[0].float().cpu()
        target_index = int(target.split(":", 1)[0]) if target else int(logits_tensor.argmax())
        masked = logits_tensor.clone()
        masked[target_index] = -self.torch.inf
        competitor_index = int(masked.argmax())
        predicted_index = int(logits_tensor.argmax())
        probabilities = logits_tensor.softmax(dim=0)
        margin = float(logits_tensor[target_index] - logits_tensor[competitor_index])
        label = self.label_for_index
        return {
            "predicted_index": predicted_index,
            "target_index": target_index,
            "predicted_class": label(predicted_index),
            "target_class": target or label(target_index),
            "target_probability": round(float(probabilities[target_index]), 5),
            "target_logit": round(float(logits_tensor[target_index]), 6),
            "competitor_class": label(competitor_index),
            "competitor_logit": round(float(logits_tensor[competitor_index]), 6),
            "margin": round(margin, 6),
            "recovered": margin > 0,
        }


def build_model():
    requested = os.getenv("EDITTRACE_MODEL", "surrogate").strip().lower()
    return SurrogateModel() if requested == "surrogate" else TorchvisionModel(requested)


MODEL = build_model()


def starter_image() -> Image.Image:
    size = 768
    image = Image.new("RGB", (size, size), "#d9e5e6")
    draw = ImageDraw.Draw(image)
    for y in range(size):
        t = y / size
        draw.line((0, y, size, y), fill=(int(211 - 45 * t), int(231 - 30 * t), int(232 - 6 * t)))
    draw.rectangle((0, 500, size, size), fill="#5d704f")
    draw.ellipse((470, 80, 630, 240), fill="#f5c56d")
    for offset in range(-160, 240, 32):
        draw.polygon([(offset, 520), (offset + 260, 255), (offset + 520, 520)], fill="#647986")
        draw.line([(offset, 520), (offset + 260, 255), (offset + 520, 520)], fill="#344650", width=6)
    draw.rectangle((262, 430, 506, 650), fill="#c04e3e", outline="#67342f", width=10)
    draw.polygon([(230, 440), (384, 320), (538, 440)], fill="#263c46")
    for x in (302, 412):
        draw.rectangle((x, 490, x + 58, 570), fill="#f4d7a4", outline="#563e36", width=7)
    draw.rectangle((350, 555, 420, 650), fill="#60433a")
    return image


def image_from_data_url(value: str) -> Image.Image:
    if "," in value:
        value = value.split(",", 1)[1]
    raw = base64.b64decode(value, validate=True)
    image = Image.open(io.BytesIO(raw))
    image.load()
    return image.convert("RGB")


def image_data_url(image: Image.Image, max_side: int = 1000) -> str:
    output = image.copy()
    output.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    output.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def validate_params(edit_type: str, params: dict[str, Any]) -> dict[str, Any]:
    if edit_type not in EDIT_SPECS:
        raise ValueError(f"Unknown edit type: {edit_type}")
    if edit_type == "crop":
        normalized = {key: float(params.get(key, default)) for key, default in {"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.8}.items()}
        if not (0 <= normalized["x"] < 1 and 0 <= normalized["y"] < 1):
            raise ValueError("Crop origin must be normalized to [0, 1).")
        if normalized["width"] <= 0 or normalized["height"] <= 0:
            raise ValueError("Crop dimensions must be positive.")
        if normalized["x"] + normalized["width"] > 1 or normalized["y"] + normalized["height"] > 1:
            raise ValueError("Crop rectangle must stay inside the image.")
        if "interpolation" in params:
            normalized["interpolation"] = str(params["interpolation"])
        return normalized
    if edit_type == "jpeg" and params.get("bypass"):
        return {"quality": 100, "bypass": True}
    spec = EDIT_SPECS[edit_type]
    key = spec["param"]
    default = {"brightness": 1.15, "contrast": 1.15, "saturation": 1.2, "blur": 2.0, "jpeg": 55}[edit_type]
    value = float(params.get(key, default))
    if not spec["min"] <= value <= spec["max"]:
        raise ValueError(f"{key} must be between {spec['min']} and {spec['max']}.")
    if edit_type == "jpeg":
        value = int(round(value))
    return {key: value}


def apply_action(image: Image.Image, action: Action, override: dict[str, Any] | None = None) -> Image.Image:
    params = override or action.params
    kind = action.type
    if kind == "brightness":
        return ImageEnhance.Brightness(image).enhance(float(params["factor"]))
    if kind == "contrast":
        return ImageEnhance.Contrast(image).enhance(float(params["factor"]))
    if kind == "saturation":
        return ImageEnhance.Color(image).enhance(float(params["factor"]))
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(float(params["radius"])))
    if kind == "crop":
        width, height = image.size
        left = int(round(params["x"] * width))
        top = int(round(params["y"] * height))
        right = int(round((params["x"] + params["width"]) * width))
        bottom = int(round((params["y"] + params["height"]) * height))
        interpolation = str(params.get("interpolation", "lanczos")).lower()
        resampling = {
            "nearest": Image.Resampling.NEAREST,
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "lanczos": Image.Resampling.LANCZOS,
        }.get(interpolation)
        if resampling is None:
            raise ValueError(f"Unsupported crop interpolation: {interpolation}")
        return image.crop((left, top, max(left + 1, right), max(top + 1, bottom))).resize(image.size, resampling)
    if kind == "jpeg":
        if params.get("bypass"):
            return image.copy()
        buffer = io.BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=int(params["quality"]),
            subsampling=int(params.get("subsampling", 2)),
            optimize=bool(params.get("optimize", False)),
            progressive=bool(params.get("progressive", False)),
        )
        buffer.seek(0)
        decoded = Image.open(buffer)
        decoded.load()
        return decoded.convert("RGB")
    raise ValueError(f"Unsupported edit: {kind}")


def replay(
    session: Session,
    remove: set[str] | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
    collect_steps: bool = False,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    remove = remove or set()
    overrides = overrides or {}
    image = session.source.copy()
    steps = []
    previous_prediction = MODEL.prediction(image, session.target_class) if collect_steps else None
    for action in session.actions:
        if action.id in remove:
            continue
        image = apply_action(image, action, overrides.get(action.id))
        if collect_steps:
            after = MODEL.prediction(image, session.target_class)
            steps.append({"action_id": action.id, "before_margin": previous_prediction["margin"], "after_margin": after["margin"]})
            previous_prediction = after
    return image, steps


def distortion(reference: Image.Image, candidate: Image.Image) -> float:
    a = np.asarray(reference.resize((128, 128)), dtype=np.float32) / 255.0
    b = np.asarray(candidate.resize((128, 128)), dtype=np.float32) / 255.0
    return round(float(np.sqrt(np.mean((a - b) ** 2))), 6)


def image_hash(image: Image.Image) -> str:
    """Canonical v1 candidate: SHA-256 over contiguous RGB pixel bytes."""
    return hashlib.sha256(np.asarray(image.convert("RGB"), dtype=np.uint8).tobytes()).hexdigest()


def param_movement(action: Action, params: dict[str, Any]) -> float:
    spec = EDIT_SPECS[action.type]
    if action.type == "crop":
        identity = {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}
        numerators = [abs(params[key] - action.params[key]) for key in identity]
        denominators = [abs(identity[key] - action.params[key]) for key in identity]
        return round(max((n / d if d else 0.0) for n, d in zip(numerators, denominators, strict=True)), 6)
    if action.type == "jpeg":
        return round((float(params["quality"]) - float(action.params["quality"])) / (100 - float(action.params["quality"])), 6)
    key = spec["param"]
    span = spec["max"] - spec["min"]
    return round(abs(float(params[key]) - float(action.params[key])) / span, 6)


def action_label(action: Action) -> str:
    if action.type == "crop":
        p = action.params
        return f"Crop {p['width']:.0%} x {p['height']:.0%}"
    key = EDIT_SPECS[action.type]["param"]
    return f"{EDIT_SPECS[action.type]['label']} {action.params[key]:g}"


def diagnose(session: Session) -> dict[str, Any]:
    started = time.perf_counter()
    full_image, steps = replay(session, collect_steps=True)
    full_prediction = MODEL.prediction(full_image, session.target_class)
    model_calls = len(session.actions) + 2
    singles: list[dict[str, Any]] = []
    step_by_id = {step["action_id"]: step for step in steps}
    for index, action in enumerate(session.actions):
        image, _ = replay(session, {action.id})
        prediction = MODEL.prediction(image, session.target_class)
        model_calls += 1
        step = step_by_id[action.id]
        spec = EDIT_SPECS[action.type]
        if action.type == "crop":
            magnitude = 1 - action.params["width"] * action.params["height"]
        elif action.type == "jpeg":
            magnitude = (100 - action.params["quality"]) / 95
        else:
            magnitude = abs(action.params[spec["param"]] - spec["identity"]) / (spec["max"] - spec["min"])
        singles.append({
            "action_id": action.id,
            "index": index,
            "label": action_label(action),
            "effect": round(prediction["margin"] - full_prediction["margin"], 6),
            "counterfactual_margin": prediction["margin"],
            "restores_target": prediction["recovered"],
            "adjacent_score_drop": round(step["before_margin"] - step["after_margin"], 6),
            "edit_magnitude": round(float(magnitude), 6),
        })
    singles.sort(key=lambda row: (-row["effect"], row["index"]))

    interactions = []
    actions = session.actions
    single_margin = {row["action_id"]: row["counterfactual_margin"] for row in singles}
    for i in range(len(actions)):
        for j in range(i + 1, len(actions)):
            a, b = actions[i], actions[j]
            joint_image, _ = replay(session, {a.id, b.id})
            joint = MODEL.prediction(joint_image, session.target_class)
            model_calls += 1
            value = joint["margin"] - single_margin[a.id] - single_margin[b.id] + full_prediction["margin"]
            interactions.append({
                "action_ids": [a.id, b.id],
                "labels": [action_label(a), action_label(b)],
                "effect": round(value, 6),
                "joint_margin": joint["margin"],
                "pair_only_recovery": bool(joint["recovered"] and single_margin[a.id] <= 0 and single_margin[b.id] <= 0),
            })
    interactions.sort(key=lambda row: (-abs(row["effect"]), row["action_ids"]))

    repairs: list[dict[str, Any]] = []
    for row in singles:
        if row["restores_target"]:
            repairs.append(_repair_record(session, "rollback", 1, [row["action_id"]], {}, row["counterfactual_margin"], distortion(full_image, replay(session, {row["action_id"]})[0]), 1.0))
    for action in actions:
        spec = EDIT_SPECS[action.type]
        for step in range(1, 21):
            fraction = step / 20
            if action.type == "crop":
                identity = {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}
                params = {key: round(action.params[key] + (value - action.params[key]) * fraction, 6) for key, value in identity.items()}
                if "interpolation" in action.params:
                    params["interpolation"] = action.params["interpolation"]
            elif action.type == "jpeg":
                quality = round(action.params["quality"] + (100 - action.params["quality"]) * fraction)
                params = {
                    "quality": quality, "bypass": step == 20,
                    "subsampling": action.params.get("subsampling", 2),
                    "optimize": action.params.get("optimize", False),
                    "progressive": action.params.get("progressive", False),
                }
            else:
                key = spec["param"]
                original = float(action.params[key])
                identity = float(spec["identity"])
                params = {key: round(original + (identity - original) * fraction, 6)}
            candidate_image, _ = replay(session, overrides={action.id: params})
            candidate_prediction = MODEL.prediction(candidate_image, session.target_class)
            model_calls += 1
            if candidate_prediction["recovered"]:
                repairs.append(_repair_record(session, "attenuate", 1, [action.id], {action.id: params}, candidate_prediction["margin"], distortion(full_image, candidate_image), param_movement(action, params)))
                break
    if not repairs:
        for pair in interactions:
            candidate_image, _ = replay(session, set(pair["action_ids"]))
            candidate_prediction = MODEL.prediction(candidate_image, session.target_class)
            model_calls += 1
            if candidate_prediction["recovered"]:
                repairs.append(_repair_record(session, "pair_rollback", 2, pair["action_ids"], {}, candidate_prediction["margin"], distortion(full_image, candidate_image), 2.0))
    repairs.sort(key=lambda row: (row["tier"], row["distortion"], row["parameter_movement"], -row["restored_margin"], row["id"]))
    elapsed = (time.perf_counter() - started) * 1000
    return {
        "prediction": full_prediction,
        "ranking": singles,
        "interactions": interactions,
        "repairs": repairs,
        "latency_ms": round(elapsed, 2),
        "model_calls": model_calls,
    }


def _repair_record(session: Session, kind: str, tier: int, remove: list[str], overrides: dict[str, Any], margin: float, image_distortion: float, movement: float) -> dict[str, Any]:
    payload = json.dumps({"kind": kind, "remove": remove, "overrides": overrides}, sort_keys=True)
    action_by_id = {action.id: (index, action) for index, action in enumerate(session.actions)}
    action_details = []
    for action_id in remove:
        index, action = action_by_id[action_id]
        detail = {
            "action_id": action_id,
            "action_number": index + 1,
            "label": action_label(action),
            "type": action.type,
            "original_params": action.params,
        }
        if action_id in overrides:
            detail["repaired_params"] = overrides[action_id]
        action_details.append(detail)
    return {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, payload)),
        "kind": kind,
        "tier": tier,
        "remove": remove,
        "overrides": overrides,
        "actions": action_details,
        "restored_margin": margin,
        "distortion": image_distortion,
        "parameter_movement": movement,
    }


def serialize_session(session: Session, include_image: bool = True) -> dict[str, Any]:
    current, _ = replay(session)
    source_prediction = MODEL.prediction(session.source, session.target_class)
    prediction = MODEL.prediction(current, session.target_class)
    result = {
        "id": session.id,
        "filename": session.filename,
        "model": {
            "id": getattr(MODEL, "weights_id", "surrogate-visual-v1"),
            "mode": "EXPERIMENTAL" if getattr(MODEL, "experimental", False) else "DEMO",
            "experimental": getattr(MODEL, "experimental", False),
        },
        "target_class": session.target_class,
        "dataset": {
            "mode": session.dataset_mode,
            "trace_id": session.trace_id,
            "benchmark": session.benchmark,
            "source_id": session.source_id,
            "ground_truth_label": session.ground_truth_label,
            "eligible": session.eligible,
            "eligibility_reason": session.eligibility_reason,
            "replay_valid": session.replay_valid,
            "replay_reason": session.replay_reason,
            "source_hash_match": session.source_hash_match,
            "action_hashes_matching": session.frozen_hashes_matching,
            "action_hashes_checked": session.frozen_hashes_checked,
        },
        "source_prediction": source_prediction,
        "prediction": prediction,
        "actions": [{**asdict(action), "label": action_label(action)} for action in session.actions],
        "can_undo": bool(session.undo),
        "edit_specs": EDIT_SPECS,
    }
    if include_image:
        result["source_image"] = image_data_url(session.source)
        result["current_image"] = image_data_url(current)
    return result


def create_session(
    image: Image.Image | None = None,
    filename: str = "edittrace-starter.png",
    ground_truth_label: int | None = None,
    **metadata: Any,
) -> Session:
    image = (image or starter_image()).convert("RGB")
    if max(image.size) > 1600 and not metadata.get("dataset_mode", False):
        image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    initial = MODEL.prediction(image)
    if ground_truth_label is not None and hasattr(MODEL, "label_for_index"):
        target = MODEL.label_for_index(int(ground_truth_label))
        eligible = initial.get("predicted_index") == int(ground_truth_label)
        reason = "source prediction matches ground truth" if eligible else "source prediction does not match ground truth"
    elif ground_truth_label is not None:
        target = initial["predicted_class"]
        eligible = False
        reason = "ground-truth eligibility requires a torchvision ImageNet model"
    else:
        target = initial["predicted_class"]
        eligible = None
        reason = "free-upload mode has no ground-truth eligibility test"
    return Session(
        id=str(uuid.uuid4()), source=image, filename=filename, target_class=target,
        ground_truth_label=ground_truth_label, eligible=eligible, eligibility_reason=reason, **metadata,
    )


def add_action(session: Session, edit_type: str, params: dict[str, Any]) -> Action:
    if len(session.actions) >= 6:
        raise ValueError("The frozen protocol limits histories to six actions.")
    if edit_type in {action.type for action in session.actions}:
        raise ValueError("The frozen protocol samples edit types without replacement.")
    normalized = validate_params(edit_type, params)
    current, _ = replay(session)
    session.undo.append(Snapshot(actions=[Action(**asdict(item)) for item in session.actions]))
    action = Action(id=str(uuid.uuid4()), type=edit_type, params=normalized, input_size=current.size, output_size=current.size)
    session.actions.append(action)
    return action


def apply_repair(session: Session, candidate: dict[str, Any]) -> None:
    session.undo.append(Snapshot(actions=[Action(**asdict(item)) for item in session.actions]))
    remove = set(candidate.get("remove", []))
    overrides = candidate.get("overrides", {})
    kept = []
    for action in session.actions:
        if action.id in remove:
            continue
        if action.id in overrides:
            params = validate_params(action.type, overrides[action.id])
            if action.type == "jpeg" and params.get("bypass"):
                continue
            action = Action(**{**asdict(action), "params": params})
        kept.append(action)
    session.actions = kept
