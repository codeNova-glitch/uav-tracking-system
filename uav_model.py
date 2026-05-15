"""
uav_model.py – UAV-specialized model utilities
===============================================
Handles:
  • Downloading a UAV-specific YOLOv8 model trained on drone datasets
  • Falling back to COCO yolov8n.pt with drone-relevant class filtering
  • Checking GPU / device availability
"""

import sys
from pathlib import Path

# UAV-relevant COCO class IDs
# 4  = airplane  (catches fixed-wing UAVs)
# 0  = person    (for context / anti-collision)
UAV_COCO_CLASSES = [0, 4]

# Known public drone detection model checkpoints on HuggingFace / GitHub
# (yolov8 format)
UAV_MODEL_SOURCES = [
    # HuggingFace – real UAV-specific weights
    "https://huggingface.co/keremberke/yolov8n-drone-detection/resolve/main/best.pt",
]


def get_device():
    """Return best available device string."""
    try:
        import torch
        if torch.cuda.is_available():
            d = f"cuda:{torch.cuda.current_device()}"
            n = torch.cuda.get_device_name()
            print(f"  🎮  GPU detected: {n}  →  using {d}")
            return d
    except ImportError:
        pass
    print("  💻  No GPU found – using CPU (may be slower)")
    return "cpu"


def download_uav_model(dest: Path = Path("models/uav_yolov8n.pt")) -> Path | None:
    """
    Try to download a pre-trained UAV/drone detection model.
    Returns the local path on success, None on failure.
    """
    import urllib.request, ssl

    dest.parent.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    for url in UAV_MODEL_SOURCES:
        try:
            print(f"  ⬇️   Downloading UAV model from:\n      {url}")
            urllib.request.urlretrieve(url, dest)
            print(f"  ✅  Saved to {dest}")
            return dest
        except Exception as e:
            print(f"  ⚠️   Failed: {e}")

    return None


def resolve_model(model_arg: str) -> tuple[str, list | None]:
    """
    Return (model_path_str, class_filter_or_None).

    Priority:
      1. If model_arg points to an existing file → use it, no class filter
      2. If model_arg == 'auto' → try downloading UAV-specific weights,
         fall back to yolov8n.pt + UAV_COCO_CLASSES filter
      3. Otherwise treat model_arg as a YOLO hub name (e.g. 'yolov8s.pt')
    """
    p = Path(model_arg)

    if p.exists():
        print(f"  📦  Using custom model: {p}")
        return str(p), None

    if model_arg == "auto":
        dest = Path("models/uav_yolov8n.pt")
        if dest.exists():
            print(f"  📦  Cached UAV model found: {dest}")
            return str(dest), None
        downloaded = download_uav_model(dest)
        if downloaded:
            return str(downloaded), None
        # Fallback
        print("  ℹ️   Falling back to yolov8n.pt with UAV class filter")
        return "yolov8n.pt", UAV_COCO_CLASSES

    # Standard YOLO model name
    return model_arg, None


if __name__ == "__main__":
    dev = get_device()
    path, cls = resolve_model("auto")
    print(f"\nModel : {path}\nDevice: {dev}\nFilter: {cls}")
