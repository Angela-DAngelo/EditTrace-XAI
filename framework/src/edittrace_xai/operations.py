from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Mapping

from PIL import Image, ImageEnhance, ImageFilter, __version__ as pillow_version

from .schema import OperationName


class OperationValidationError(ValueError):
    """Raised when an edit parameterization violates the frozen contract."""


RESAMPLING = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


@dataclass(frozen=True)
class OperationOutput:
    image: Image.Image
    codec_metadata: dict[str, Any] | None = None


def ensure_rgb_uint8(image: Image.Image) -> Image.Image:
    return image if image.mode == "RGB" else image.convert("RGB")


def _positive_factor(params: Mapping[str, Any], key: str = "factor") -> float:
    value = float(params[key])
    if value < 0:
        raise OperationValidationError(f"{key} must be non-negative")
    return value


def apply_operation(
    image: Image.Image,
    op_name: OperationName,
    params: Mapping[str, Any],
) -> OperationOutput:
    image = ensure_rgb_uint8(image)

    if op_name == OperationName.BRIGHTNESS:
        factor = _positive_factor(params)
        return OperationOutput(ImageEnhance.Brightness(image).enhance(factor))

    if op_name == OperationName.CONTRAST:
        factor = _positive_factor(params)
        return OperationOutput(ImageEnhance.Contrast(image).enhance(factor))

    if op_name == OperationName.SATURATION:
        factor = _positive_factor(params)
        return OperationOutput(ImageEnhance.Color(image).enhance(factor))

    if op_name == OperationName.GAUSSIAN_BLUR:
        sigma = float(params["sigma"])
        if sigma < 0:
            raise OperationValidationError("sigma must be non-negative")
        if bool(params.get("bypass", False)) or sigma == 0:
            return OperationOutput(image.copy())
        return OperationOutput(image.filter(ImageFilter.GaussianBlur(radius=sigma)))

    if op_name == OperationName.CROP_RESIZE:
        box = params["box"]
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise OperationValidationError("crop box must contain four normalized coordinates")
        left, top, right, bottom = map(float, box)
        if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
            raise OperationValidationError("crop box must be non-empty and within [0,1]")
        interpolation = str(params.get("interpolation", "bicubic"))
        if interpolation not in RESAMPLING:
            raise OperationValidationError(f"unsupported interpolation: {interpolation}")
        width, height = image.size
        pixel_box = (
            max(0, min(width - 1, round(left * width))),
            max(0, min(height - 1, round(top * height))),
            max(1, min(width, round(right * width))),
            max(1, min(height, round(bottom * height))),
        )
        if pixel_box[2] <= pixel_box[0] or pixel_box[3] <= pixel_box[1]:
            raise OperationValidationError("crop box becomes empty after pixel rounding")
        cropped = image.crop(pixel_box)
        resized = cropped.resize((width, height), resample=RESAMPLING[interpolation])
        return OperationOutput(resized)

    if op_name == OperationName.JPEG:
        if bool(params.get("bypass", False)):
            return OperationOutput(
                image.copy(),
                codec_metadata={
                    "library": "Pillow",
                    "library_version": pillow_version,
                    "quality": None,
                    "subsampling": int(params.get("subsampling", 0)),
                    "optimize": bool(params.get("optimize", False)),
                    "progressive": bool(params.get("progressive", False)),
                    "bypass": True,
                },
            )
        quality = int(params["quality"])
        if not 1 <= quality <= 100:
            raise OperationValidationError("JPEG quality must be in [1,100]")
        subsampling = int(params.get("subsampling", 0))
        optimize = bool(params.get("optimize", False))
        progressive = bool(params.get("progressive", False))
        buffer = io.BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=quality,
            subsampling=subsampling,
            optimize=optimize,
            progressive=progressive,
        )
        buffer.seek(0)
        decoded = Image.open(buffer).convert("RGB")
        decoded.load()
        return OperationOutput(
            decoded,
            codec_metadata={
                "library": "Pillow",
                "library_version": pillow_version,
                "quality": quality,
                "subsampling": subsampling,
                "optimize": optimize,
                "progressive": progressive,
                "bypass": False,
            },
        )

    raise OperationValidationError(f"unsupported operation: {op_name}")


def identity_params(op_name: OperationName, params: Mapping[str, Any]) -> dict[str, Any]:
    if op_name in {
        OperationName.BRIGHTNESS,
        OperationName.CONTRAST,
        OperationName.SATURATION,
    }:
        return {"factor": 1.0}
    if op_name == OperationName.GAUSSIAN_BLUR:
        return {"sigma": 0.0, "bypass": True}
    if op_name == OperationName.CROP_RESIZE:
        return {
            "box": [0.0, 0.0, 1.0, 1.0],
            "interpolation": str(params.get("interpolation", "bicubic")),
        }
    if op_name == OperationName.JPEG:
        return {
            "quality": 100,
            "subsampling": int(params.get("subsampling", 0)),
            "optimize": bool(params.get("optimize", False)),
            "progressive": bool(params.get("progressive", False)),
            "bypass": True,
        }
    raise OperationValidationError(f"unsupported operation: {op_name}")


def attenuate_params(
    op_name: OperationName,
    params: Mapping[str, Any],
    alpha: float,
) -> dict[str, Any]:
    if not 0 <= alpha <= 1:
        raise OperationValidationError("attenuation alpha must be in [0,1]")
    identity = identity_params(op_name, params)

    if op_name in {
        OperationName.BRIGHTNESS,
        OperationName.CONTRAST,
        OperationName.SATURATION,
    }:
        start = float(params["factor"])
        return {"factor": (1 - alpha) * start + alpha}

    if op_name == OperationName.GAUSSIAN_BLUR:
        sigma = (1 - alpha) * float(params["sigma"])
        return {"sigma": sigma, "bypass": alpha == 1 or sigma == 0}

    if op_name == OperationName.CROP_RESIZE:
        start = list(map(float, params["box"]))
        target = identity["box"]
        box = [(1 - alpha) * a + alpha * b for a, b in zip(start, target, strict=True)]
        return {"box": box, "interpolation": identity["interpolation"]}

    if op_name == OperationName.JPEG:
        start = int(params["quality"])
        result = dict(identity)
        result["quality"] = round((1 - alpha) * start + alpha * 100)
        result["bypass"] = alpha == 1
        return result

    raise OperationValidationError(f"unsupported operation: {op_name}")

