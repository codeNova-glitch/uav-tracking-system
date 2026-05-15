# 🛸 UAV Real-Time Detection & Tracking System

> Frame-by-frame drone detection with motion trails, ID persistence, and annotated video output.  
> **Two complementary engines** — YOLO+ByteTrack (deep learning) and MOG2 background subtraction (no GPU needed).

---

## ✨ Features

| Feature | YOLO Tracker | BG-Subtraction Tracker |
|---|---|---|
| Deep-learning detection | ✅ YOLOv8 | ❌ (classic CV) |
| GPU acceleration | ✅ CUDA/MPS | ❌ CPU only |
| Works without trained model | ❌ | ✅ |
| Best for | High-resolution, complex scenes | Sky/static backgrounds |
| Tracking algorithm | ByteTrack (built-in) | Centroid matching |
| Motion trail overlay | ✅ | ✅ |
| Unique ID per drone | ✅ | ✅ |
| HUD overlay | ✅ | ✅ |

---

## 📦 Installation

```bash
# Clone / download the project folder, then:
pip install -r requirements.txt
```

GPU support (optional):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 🚀 Quick Start

### 1 – Generate a synthetic test video (no drone video? no problem)
```bash
python generate_test_video.py --drones 3 --seconds 10
# → test_input.mp4
```

### 2a – YOLO + ByteTrack detector (recommended)
```bash
# Auto-select best available model
python uav_tracker.py --input test_input.mp4

# Specify a confidence threshold
python uav_tracker.py --input drone_footage.mp4 --conf 0.3

# Use a larger, more accurate model
python uav_tracker.py --input drone_footage.mp4 --model yolov8m.pt

# Show live preview while processing
python uav_tracker.py --input drone_footage.mp4 --show
```

### 2b – Background-subtraction tracker (no GPU, great for sky footage)
```bash
python uav_bg_tracker.py --input test_input.mp4

# Tune sensitivity (smaller = detect tinier drones)
python uav_bg_tracker.py --input sky_footage.mp4 --min-area 40
```

### Output
Both scripts write an annotated `.mp4` beside the input file:
```
drone_footage.mp4  →  drone_footage_tracked.mp4
test_input.mp4     →  test_input_tracked.mp4        (YOLO)
test_input.mp4     →  test_input_bg_tracked.mp4     (BG-sub)
```

---

## ⚙️ CLI Reference

### `uav_tracker.py` (YOLO + ByteTrack)

| Flag | Default | Description |
|---|---|---|
| `--input` | *(required)* | Path to input video |
| `--output` | auto | Output video path |
| `--model` | `yolov8n.pt` | YOLOv8 weights file |
| `--conf` | `0.25` | Detection confidence threshold |
| `--iou` | `0.45` | NMS IoU threshold |
| `--classes` | all | COCO class IDs (comma-sep, e.g. `0,4`) |
| `--trail-len` | `40` | Motion trail length in frames |
| `--show` | off | Show live preview window |
| `--hide-conf` | off | Hide confidence scores in labels |

### `uav_bg_tracker.py` (MOG2)

| Flag | Default | Description |
|---|---|---|
| `--input` | *(required)* | Path to input video |
| `--output` | auto | Output video path |
| `--min-area` | `80` | Min blob area in pixels² |
| `--max-area` | `8000` | Max blob area in pixels² |

### `generate_test_video.py`

| Flag | Default | Description |
|---|---|---|
| `--out` | `test_input.mp4` | Output path |
| `--drones` | `3` | Number of synthetic drones |
| `--seconds` | `10` | Video duration |
| `--fps` | `25` | Frames per second |
| `--width` | `1280` | Frame width |
| `--height` | `720` | Frame height |

---

## 🎯 Model Selection Guide

| Scenario | Recommended model |
|---|---|
| Fast CPU-only | `yolov8n.pt` (nano) |
| GPU available | `yolov8s.pt` or `yolov8m.pt` |
| Maximum accuracy | `yolov8x.pt` |
| UAV-specific dataset | Custom `.pt` trained on [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) |
| Static sky background | `uav_bg_tracker.py` (no model needed) |

---

## 📊 Visual Annotations Explained

```
┌──────────────────────────────────────────────────────┐
│ Frame 142/750  |  18.9%  |  FPS: 24.1  |  Tracks: 2 │  ← HUD bar
├──────────────────────────────────────────────────────┤
│                                                      │
│    ┌──┐ID:1 drone 0.87         ← Label + confidence  │
│   ╔╪══╪╗   ← Corner ticks (military style)          │
│   ╠╪  ╪╣                                             │
│   ╚╪══╪╝                                             │
│    └──┘                                              │
│   ~~~/    ← Colour-fading motion trail               │
│                                                      │
└──────────────────────────────────────────────────────┘
            UAV-TRACKER  |  YOLOv8 + ByteTrack         ← Watermark
```

---

## 🔧 Tips & Tricks

- **Small drones far away**: lower `--conf` to `0.15`, increase `--trail-len` to `60`
- **Many false positives**: raise `--conf` to `0.4`+, or pass `--classes 4` (COCO airplane class)
- **Night / IR footage**: use BG-subtraction tracker with `--min-area 30`
- **Very high-res 4K video**: resize before processing or use `--model yolov8n.pt` for speed
- **Custom trained model**: use `python uav_tracker.py --model my_uav_model.pt`

---

## 🗂️ Project Structure

```
uav_tracker/
├── uav_tracker.py          # Main YOLO + ByteTrack pipeline
├── uav_bg_tracker.py       # Background-subtraction pipeline
├── uav_model.py            # Model resolution / download helper
├── generate_test_video.py  # Synthetic test video generator
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 📚 References

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864)
- [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset) – largest public UAV detection benchmark
- [OpenCV MOG2 Background Subtractor](https://docs.opencv.org/4.x/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html)
