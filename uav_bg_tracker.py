"""
uav_bg_tracker.py  –  Background-subtraction UAV tracker (no ML model needed)
==============================================================================
Uses MOG2 background subtraction + contour filtering + centroid tracking.
Perfect for detecting small/fast-moving drones against a static or slow
panning sky background.  Works as a strong baseline when no trained model
is available, and pairs well with the YOLO tracker for hard-to-detect objects.

Usage:
    python uav_bg_tracker.py --input video.mp4
    python uav_bg_tracker.py --input video.mp4 --output tracked_bg.mp4 --min-area 60
"""

import argparse
import time
from pathlib import Path
from collections import defaultdict, deque

import cv2
import numpy as np


# ── Simple centroid tracker ────────────────────────────────────────────────────
class CentroidTracker:
    def __init__(self, max_lost=10, max_dist=80):
        self.next_id  = 0
        self.objects  = {}   # id → (cx, cy)
        self.lost     = defaultdict(int)
        self.max_lost = max_lost
        self.max_dist = max_dist

    def update(self, centroids):
        if not centroids:
            for oid in list(self.lost):
                self.lost[oid] += 1
                if self.lost[oid] > self.max_lost:
                    del self.objects[oid]
                    del self.lost[oid]
            return dict(self.objects)

        if not self.objects:
            for c in centroids:
                self.objects[self.next_id] = c
                self.next_id += 1
            return dict(self.objects)

        obj_ids  = list(self.objects.keys())
        obj_cxys = list(self.objects.values())
        used_obj = set()
        used_cen = set()

        # Greedy nearest-centroid match
        dists = []
        for ci, c in enumerate(centroids):
            for oi, o in enumerate(obj_cxys):
                d = np.linalg.norm(np.array(c) - np.array(o))
                dists.append((d, oi, ci))
        dists.sort()

        for d, oi, ci in dists:
            if oi in used_obj or ci in used_cen:
                continue
            if d > self.max_dist:
                break
            oid = obj_ids[oi]
            self.objects[oid] = centroids[ci]
            self.lost[oid]    = 0
            used_obj.add(oi)
            used_cen.add(ci)

        # Register new
        for ci, c in enumerate(centroids):
            if ci not in used_cen:
                self.objects[self.next_id] = c
                self.next_id += 1

        # Age un-matched
        for oi, oid in enumerate(obj_ids):
            if oi not in used_obj:
                self.lost[oid] += 1
                if self.lost[oid] > self.max_lost:
                    del self.objects[oid]
                    del self.lost[oid]

        return dict(self.objects)


PALETTE = [
    (255, 56,  56), (0,  194, 255), (72,  249, 10), (255, 178, 29),
    (132, 56,  255), (203, 56, 255), (255, 112, 31), (44, 153, 168),
]

def colour(tid):
    return PALETTE[int(tid) % len(PALETTE)]


def run(args):
    in_path  = Path(args.input)
    out_path = Path(args.output) if args.output else \
               in_path.with_name(in_path.stem + "_bg_tracked.mp4")

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {in_path}")

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0
    TOT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer  = cv2.VideoWriter(str(out_path),
                              cv2.VideoWriter_fourcc(*"mp4v"),
                              FPS, (W, H))
    fgbg    = cv2.createBackgroundSubtractorMOG2(
                  history=200, varThreshold=40, detectShadows=False)
    tracker = CentroidTracker(max_lost=15, max_dist=60)
    trails  = defaultdict(lambda: deque(maxlen=40))

    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    frame_idx = 0
    t_start   = time.time()

    print(f"\n🚀  BG-subtraction tracker  →  {out_path}\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # ── BG subtraction + morphology ──
        fg = fgbg.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
        fg = cv2.dilate(fg, kernel, iterations=2)

        # ── Contour detection ──
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centroids = []
        boxes_out  = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < args.min_area or area > args.max_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            ar = w / (h + 1e-6)
            if ar > 6 or ar < 0.17:   # reject very elongated shapes (trees, wires)
                continue
            cx, cy = x + w // 2, y + h // 2
            centroids.append((cx, cy))
            boxes_out.append((x, y, x + w, y + h))

        tracked = tracker.update(centroids)

        # ── Draw ──
        out = frame.copy()
        for tid, (cx, cy) in tracked.items():
            trails[tid].append((cx, cy))
            col = colour(tid)
            # Find matching bbox
            best_box = None
            best_d   = 9999
            for (x1, y1, x2, y2) in boxes_out:
                bx, by = (x1+x2)//2, (y1+y2)//2
                d = abs(bx - cx) + abs(by - cy)
                if d < best_d:
                    best_d, best_box = d, (x1, y1, x2, y2)
            if best_box:
                x1, y1, x2, y2 = best_box
                cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
                # Corner ticks
                t = 8
                for (px, py), (dx, dy) in [
                    ((x1,y1),(1,1)),((x2,y1),(-1,1)),
                    ((x1,y2),(1,-1)),((x2,y2),(-1,-1))]:
                    cv2.line(out,(px,py),(px+dx*t,py),col,2)
                    cv2.line(out,(px,py),(px,py+dy*t),col,2)

            label = f"UAV ID:{tid}"
            cv2.putText(out, label, (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            cv2.drawMarker(out, (cx, cy), col, cv2.MARKER_CROSS, 10, 1)

            # Trail
            pts = list(trails[tid])
            for i in range(1, len(pts)):
                a = i / len(pts)
                tc = tuple(int(c * a) for c in col)
                cv2.line(out, pts[i-1], pts[i], tc, max(1,int(a*3)), cv2.LINE_AA)

        # Remove stale trails
        for tid in set(trails) - set(tracked):
            del trails[tid]

        # HUD
        pct = frame_idx / max(TOT, 1) * 100
        elapsed = time.time() - t_start
        fps_proc = frame_idx / elapsed
        overlay = out.copy()
        cv2.rectangle(overlay, (0,0), (W, 36), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.5, out, 0.5, 0, out)
        hud = (f"Frame {frame_idx}/{TOT}  |  {pct:.1f}%  |  "
               f"FPS:{fps_proc:.1f}  |  Tracks:{len(tracked)}")
        cv2.putText(out, hud, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,255,120), 1, cv2.LINE_AA)
        bar_w = int(W * frame_idx / max(TOT,1))
        cv2.rectangle(out, (0,34), (bar_w,36), (0,220,80), -1)
        wm = "UAV-TRACKER  |  MOG2 BG-Subtraction"
        (ww,wh),_ = cv2.getTextSize(wm, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.putText(out, wm, (W-ww-8, H-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,180), 1, cv2.LINE_AA)

        writer.write(out)

        if frame_idx % 25 == 0:
            bar = "█" * int(pct/5) + "░" * (20-int(pct/5))
            print(f"  [{bar}] {pct:5.1f}%  tracks:{len(tracked)}", end="\r")

    cap.release()
    writer.release()
    elapsed = time.time() - t_start
    print(f"\n\n  ✅  Done in {elapsed:.1f}s  →  {out_path}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",    required=True)
    ap.add_argument("--output",   default=None)
    ap.add_argument("--min-area", type=int, default=80,
                    help="Min contour area in pixels (default 80)")
    ap.add_argument("--max-area", type=int, default=8000,
                    help="Max contour area in pixels (default 8000)")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
