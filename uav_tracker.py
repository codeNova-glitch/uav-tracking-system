"""
UAV (Drone) Real-Time Detection & Tracking System
==================================================
Uses YOLOv8 for detection + ByteTrack for multi-object tracking.
Processes video frame-by-frame and outputs an annotated video.

Usage:
    python uav_tracker.py --input <video_path> [options]

Examples:
    python uav_tracker.py --input flight.mp4
    python uav_tracker.py --input drone_footage.mp4 --conf 0.3 --model yolov8s.pt
    python uav_tracker.py --input cam.mp4 --output result.mp4 --show
"""

import argparse
import sys
import time
from pathlib import Path
from collections import defaultdict, deque
import cv2
import numpy as np

# ── Colour palette for track IDs ──────────────────────────────────────────────
PALETTE = [
    (255, 56,  56),  (255, 157,  151), (255, 112,  31),  (255, 178,  29),
    (207, 210,  49),  (72,  249, 10),  (146, 204,  23),  (61,  219, 134),
    (26,  147, 52),  (0,   212, 187),  (44,  153, 168),  (0,   194, 255),
    (52,  69,  147),  (100,  115, 255), (0,   24,  236),  (132, 56,  255),
    (82,  0,   133),  (203, 56,  255),  (255, 149, 200),  (255, 55,  199),
]

def get_colour(track_id: int):
    return PALETTE[int(track_id) % len(PALETTE)]


# ── Trail / history ────────────────────────────────────────────────────────────
class TrailBuffer:
    """Stores centre-point history per track for drawing motion trails."""
    def __init__(self, maxlen: int = 40):
        self.trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=maxlen))

    def update(self, track_id: int, cx: int, cy: int):
        self.trails[track_id].append((cx, cy))

    def get(self, track_id: int):
        return list(self.trails[track_id])

    def prune(self, active_ids: set):
        dead = set(self.trails) - active_ids
        for tid in dead:
            del self.trails[tid]


# ── Frame annotator ────────────────────────────────────────────────────────────
def draw_detections(frame: np.ndarray,
                    boxes,          # xyxy tensors
                    track_ids,
                    confs,
                    class_names,
                    cls_ids,
                    trail_buf: TrailBuffer,
                    show_conf: bool = True) -> np.ndarray:
    h, w = frame.shape[:2]
    active_ids = set()

    for box, tid, conf, cid in zip(boxes, track_ids, confs, cls_ids):
        x1, y1, x2, y2 = map(int, box)
        tid  = int(tid)
        conf = float(conf)
        label_name = class_names[int(cid)] if class_names else "UAV"
        colour = get_colour(tid)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        trail_buf.update(tid, cx, cy)
        active_ids.add(tid)

        # ── bounding box ──
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        # ── corner ticks (military-style) ──
        tick = 10
        for (px, py), (dx, dy) in [
            ((x1, y1), (1, 1)),  ((x2, y1), (-1, 1)),
            ((x1, y2), (1, -1)), ((x2, y2), (-1, -1)),
        ]:
            cv2.line(frame, (px, py), (px + dx * tick, py), colour, 3)
            cv2.line(frame, (px, py), (px, py + dy * tick), colour, 3)

        # ── crosshair at centre ──
        cv2.drawMarker(frame, (cx, cy), colour,
                       cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA)

        # ── label ──
        text = f"ID:{tid} {label_name}"
        if show_conf:
            text += f" {conf:.2f}"
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty = max(y1 - 6, th + 4)
        cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 4, ty + bl), colour, -1)
        cv2.putText(frame, text, (x1 + 2, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # ── motion trails ──
    for tid in active_ids:
        pts = trail_buf.get(tid)
        colour = get_colour(tid)
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            t_col = tuple(int(c * alpha) for c in colour)
            thickness = max(1, int(alpha * 3))
            cv2.line(frame, pts[i - 1], pts[i], t_col, thickness, cv2.LINE_AA)

    trail_buf.prune(active_ids)
    return frame


# ── HUD overlay ────────────────────────────────────────────────────────────────
def draw_hud(frame: np.ndarray,
             frame_idx: int,
             total_frames: int,
             fps: float,
             n_tracks: int,
             elapsed: float) -> np.ndarray:
    h, w = frame.shape[:2]

    # semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    pct = frame_idx / max(total_frames, 1) * 100
    hud = (f"Frame {frame_idx}/{total_frames}  |  "
           f"{pct:.1f}%  |  FPS: {fps:.1f}  |  "
           f"Tracks: {n_tracks}  |  Elapsed: {elapsed:.1f}s")
    cv2.putText(frame, hud, (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 120), 1, cv2.LINE_AA)

    # progress bar
    bar_w = int(w * frame_idx / max(total_frames, 1))
    cv2.rectangle(frame, (0, 34), (bar_w, 36), (0, 220, 80), -1)

    # bottom-right watermark
    wm = "UAV-TRACKER  |  YOLOv8 + ByteTrack"
    (ww, wh), _ = cv2.getTextSize(wm, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.putText(frame, wm, (w - ww - 8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
    return frame


# ── Stats accumulator ──────────────────────────────────────────────────────────
class Stats:
    def __init__(self):
        self.total_detections = 0
        self.frames_with_detections = 0
        self.unique_ids: set = set()
        self.max_simultaneous = 0
        self.frame_times: list = []

    def update(self, n_dets: int, track_ids, t_frame: float):
        self.total_detections += n_dets
        if n_dets:
            self.frames_with_detections += 1
        self.unique_ids.update(int(i) for i in track_ids)
        self.max_simultaneous = max(self.max_simultaneous, n_dets)
        self.frame_times.append(t_frame)

    def summary(self):
        avg_fps = 1 / (np.mean(self.frame_times) + 1e-9)
        return {
            "total_detections"      : self.total_detections,
            "frames_with_detections": self.frames_with_detections,
            "unique_track_ids"      : len(self.unique_ids),
            "max_simultaneous"      : self.max_simultaneous,
            "avg_processing_fps"    : round(avg_fps, 2),
        }


# ── Main pipeline ──────────────────────────────────────────────────────────────
def run(args):
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("❌  ultralytics not installed. Run:  pip install ultralytics")

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"❌  Input video not found: {input_path}")

    # ── auto output name ──
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = input_path.with_name(input_path.stem + "_tracked.mp4")

    print(f"\n{'═'*58}")
    print(f"  UAV Detection & Tracking System")
    print(f"{'═'*58}")
    print(f"  Input  : {input_path}")
    print(f"  Output : {out_path}")
    print(f"  Model  : {args.model}")
    print(f"  Conf   : {args.conf}   IOU: {args.iou}")
    print(f"{'═'*58}\n")

    # ── load model ──
    print("⏳  Loading YOLO model …")
    model = YOLO(args.model)

    # ── open video ──
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        sys.exit(f"❌  Cannot open video: {input_path}")

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
    TOT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Resolution : {W}×{H}   FPS: {FPS:.2f}   Frames: {TOT}")

    # ── video writer ──
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, FPS, (W, H))

    trail_buf = TrailBuffer(maxlen=args.trail_len)
    stats     = Stats()
    t_start   = time.time()
    frame_idx = 0

    # determine classes to filter (COCO: 'airplane'=4, default=all)
    class_filter = None
    if args.classes:
        class_filter = [int(c) for c in args.classes.split(",")]

    print("\n🚀  Processing …  (Ctrl+C to abort)\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            t0 = time.time()

            # ── YOLO track (ByteTrack built-in) ──
            results = model.track(
                frame,
                persist    = True,
                conf       = args.conf,
                iou        = args.iou,
                tracker    = "bytetrack.yaml",
                classes    = class_filter,
                verbose    = False,
            )

            res = results[0]

            # ── extract boxes / ids / confs ──
            boxes = ids = confs = cls_ids = []
            if res.boxes is not None and res.boxes.id is not None:
                boxes   = res.boxes.xyxy.cpu().numpy()
                ids     = res.boxes.id.cpu().numpy()
                confs   = res.boxes.conf.cpu().numpy()
                cls_ids = res.boxes.cls.cpu().numpy()

            t_frame = time.time() - t0
            stats.update(len(boxes), ids, t_frame)

            # ── annotate ──
            frame = draw_detections(
                frame, boxes, ids, confs,
                model.names, cls_ids,
                trail_buf,
                show_conf=not args.hide_conf,
            )
            frame = draw_hud(
                frame, frame_idx, TOT,
                1 / (t_frame + 1e-9),
                len(ids), time.time() - t_start,
            )

            writer.write(frame)

            if args.show:
                cv2.imshow("UAV Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n⚠️  Interrupted by user.")
                    break

            # console progress
            if frame_idx % 30 == 0 or frame_idx == 1:
                pct = frame_idx / max(TOT, 1) * 100
                elapsed = time.time() - t_start
                fps_proc = frame_idx / elapsed
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                print(f"  [{bar}] {pct:5.1f}%  frame {frame_idx}/{TOT}"
                      f"  {fps_proc:.1f} fps  tracks:{len(ids)}", end="\r")

    except KeyboardInterrupt:
        print("\n⚠️  Aborted.")

    finally:
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    # ── summary ──
    elapsed = time.time() - t_start
    s = stats.summary()
    print(f"\n\n{'═'*58}")
    print(f"  ✅  Done in {elapsed:.1f}s")
    print(f"{'─'*58}")
    print(f"  Frames processed     : {frame_idx}")
    print(f"  Frames with UAVs     : {s['frames_with_detections']}")
    print(f"  Total detections     : {s['total_detections']}")
    print(f"  Unique track IDs     : {s['unique_track_ids']}")
    print(f"  Max simultaneous     : {s['max_simultaneous']}")
    print(f"  Avg processing FPS   : {s['avg_processing_fps']}")
    print(f"  Output saved to      : {out_path}")
    print(f"{'═'*58}\n")
    return str(out_path)


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Real-time UAV detection & tracking in video files",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--input",     required=True,          help="Path to input video")
    p.add_argument("--output",    default=None,           help="Output path (default: <input>_tracked.mp4)")
    p.add_argument("--model",     default="yolov8n.pt",   help="YOLO model weights (yolov8n/s/m/l/x.pt or custom .pt)")
    p.add_argument("--conf",      type=float, default=0.25, help="Detection confidence threshold [0-1]")
    p.add_argument("--iou",       type=float, default=0.45, help="NMS IoU threshold [0-1]")
    p.add_argument("--classes",   default=None,           help="Comma-separated COCO class IDs to detect (e.g. '0,4'). Default=all")
    p.add_argument("--trail-len", type=int,   default=40, help="Trail history length (frames)")
    p.add_argument("--show",      action="store_true",    help="Show live preview window (requires display)")
    p.add_argument("--hide-conf", action="store_true",    help="Hide confidence scores in labels")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
