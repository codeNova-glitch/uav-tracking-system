# 🛸 UAV Real-Time Detection & Tracking System

> Frame-by-frame drone detection with motion trails, ID persistence, and annotated video output.  
> Two complementary engines** — YOLO+ByteTrack (deep learning) and MOG2 background subtraction (no GPU needed).

---


## Features

- Real-time UAV/drone detection and tracking
- Frame-by-frame video processing
- YOLOv8 + ByteTrack based AI tracking
- Background subtraction based tracking system
- GPU acceleration support (CUDA/MPS)
- Motion trail visualization
- Unique ID assignment for detected drones
- FPS and tracking information overlay
- Annotated output video generation
- Synthetic drone video generator for testing
- Supports high-resolution drone footage
- Command-line configurable parameters
- Automatic output video saving
- Live preview while processing





## 📊 Visual Annotation
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



## 📚 References

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864)
- [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset) – largest public UAV detection benchmark
- [OpenCV MOG2 Background Subtractor](https://docs.opencv.org/4.x/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html)
