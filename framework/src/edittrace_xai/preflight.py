from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path
from typing import Any

from PIL import __version__ as pillow_version
from pydantic import __version__ as pydantic_version

from .generators import load_config


def preflight_report(
    *,
    config_path: str | Path,
    data_root: str | Path | None = None,
    layout: str = "imagefolder",
    labels_file: str | Path | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pillow": pillow_version,
        "pydantic": pydantic_version,
        "config_schema": config["schema_version"],
        "global_seed": config["global_seed"],
        "torch_installed": importlib.util.find_spec("torch") is not None,
        "torchvision_installed": importlib.util.find_spec("torchvision") is not None,
        "cuda_available": False,
        "dataset": {"provided": data_root is not None, "layout": layout},
        "blockers": [],
    }
    if report["torch_installed"]:
        import torch

        report["torch_version"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        if report["cuda_available"]:
            report["cuda_device"] = torch.cuda.get_device_name(0)
    else:
        report["blockers"].append("torch is not installed")
    if not report["torchvision_installed"]:
        report["blockers"].append("torchvision is not installed")

    if data_root is None:
        report["blockers"].append("dataset root was not provided")
    else:
        root = Path(data_root).resolve()
        report["dataset"]["root"] = str(root)
        report["dataset"]["exists"] = root.is_dir()
        if not root.is_dir():
            report["blockers"].append("dataset root does not exist")
        elif layout == "imagefolder":
            class_dirs = [path for path in root.iterdir() if path.is_dir()]
            report["dataset"]["class_directories"] = len(class_dirs)
            if len(class_dirs) != 1000:
                report["blockers"].append(
                    f"ImageFolder layout has {len(class_dirs)} class directories, expected 1000"
                )
        elif layout == "flat":
            images = [
                path for path in root.iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
            report["dataset"]["images"] = len(images)
            if len(images) != 50000:
                report["blockers"].append(
                    f"flat validation layout has {len(images)} images, expected 50000"
                )
            if labels_file is None or not Path(labels_file).is_file():
                report["blockers"].append("flat layout requires a valid ground-truth labels file")
        else:
            report["blockers"].append(f"unsupported layout: {layout}")
    report["passed"] = not report["blockers"]
    return report

