"""
generate_test_video.py – creates a synthetic drone-scene test video
====================================================================
Renders a sky-like background with moving "drone" objects (small dark
quadcopter silhouettes) so the tracker can be demonstrated without a
real input file.

Usage:
    python generate_test_video.py              # produces test_input.mp4
    python generate_test_video.py --out myvid.mp4 --drones 4 --seconds 12
"""

import argparse
import math
import random
import cv2
import numpy as np
from pathlib import Path


# ── Drone silhouette (tiny cross + rotors) ────────────────────────────────────
def draw_drone(img, cx, cy, size=18, angle=0.0, colour=(30, 30, 30)):
    s  = size
    rs = max(4, size // 3)   # rotor radius
    # Body cross
    pts = np.array([
        [cx - s, cy],
        [cx + s, cy],
        [cx,     cy],
        [cx,     cy - s],
        [cx,     cy + s],
    ], dtype=np.int32)
    # Rotate
    rot = cv2.getRotationMatrix2D((cx, cy), math.degrees(angle), 1.0)
    def r(p):
        v = rot @ np.array([p[0], p[1], 1])
        return int(v[0]), int(v[1])

    for i in range(0, len(pts) - 1, 2):
        cv2.line(img, r(pts[i]), r(pts[i+1]), colour, 2, cv2.LINE_AA)

    # Rotors at four arm tips
    for dx, dy in [(s, 0), (-s, 0), (0, s), (0, -s)]:
        rx, ry = r((cx + dx, cy + dy))
        cv2.circle(img, (rx, ry), rs, colour, 1, cv2.LINE_AA)

    # Centre body dot
    cv2.circle(img, (cx, cy), max(3, size // 5), colour, -1, cv2.LINE_AA)


# ── Sky + cloud background ─────────────────────────────────────────────────────
def make_sky_bg(W, H):
    """Create a gradient sky background."""
    sky = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        t = y / H
        r = int(135 + (200 - 135) * t)
        g = int(206 + (220 - 206) * t)
        b = int(235 + (255 - 235) * t)
        sky[y, :] = [b, g, r]

    # Add some rough cloud patches
    rng = np.random.default_rng(42)
    for _ in range(30):
        cx = rng.integers(0, W)
        cy = rng.integers(0, H // 2)
        axes = (rng.integers(40, 120), rng.integers(20, 60))
        alpha = rng.uniform(0.05, 0.2)
        overlay = sky.copy()
        cv2.ellipse(overlay, (cx, cy), axes, 0, 0, 360, (255, 255, 255), -1)
        cv2.addWeighted(overlay, alpha, sky, 1 - alpha, 0, sky)
    return sky


# ── Parametric drone paths ─────────────────────────────────────────────────────
class DronePath:
    """Smooth Lissajous / elliptical path for a drone."""
    def __init__(self, W, H, seed=0):
        rng = random.Random(seed)
        self.cx   = rng.uniform(0.15, 0.85) * W
        self.cy   = rng.uniform(0.15, 0.85) * H
        self.rx   = rng.uniform(0.05, 0.30) * W
        self.ry   = rng.uniform(0.05, 0.20) * H
        self.wx   = rng.uniform(0.3, 1.2)
        self.wy   = rng.uniform(0.3, 1.2)
        self.phi  = rng.uniform(0, 2 * math.pi)
        self.size = rng.randint(14, 26)
        self.speed = rng.uniform(0.5, 1.5)
        # Random entry frame so drones don't all start at t=0
        self.start = rng.randint(0, 60)
        self.end_frame = None   # set later if drone exits early

    def pos(self, t):
        t *= self.speed
        x = self.cx + self.rx * math.cos(self.wx * t + self.phi)
        y = self.cy + self.ry * math.sin(self.wy * t)
        return int(x), int(y)

    def angle(self, t):
        t *= self.speed
        return t * 0.8


def generate(out_path: str, W=1280, H=720, fps=25, seconds=10, n_drones=3):
    total = fps * seconds
    bg    = make_sky_bg(W, H)
    drones = [DronePath(W, H, seed=i) for i in range(n_drones)]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    for f in range(total):
        frame = bg.copy()
        t = f / fps

        # Add subtle wind-texture noise
        noise = np.random.randint(0, 6, (H, W, 3), dtype=np.uint8)
        cv2.add(frame, noise, frame)

        for i, d in enumerate(drones):
            if f < d.start:
                continue
            px, py = d.pos(t)
            angle  = d.angle(t)

            # Shadow
            draw_drone(frame, px + 4, py + 4, d.size, angle, (60, 60, 60))
            # Drone
            draw_drone(frame, px, py, d.size, angle, (15, 15, 15))

        # Frame counter watermark
        cv2.putText(frame, f"SYNTHETIC TEST  frame {f+1}/{total}",
                    (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)

        writer.write(frame)

        if (f + 1) % (fps * 2) == 0:
            print(f"  Generated {f+1}/{total} frames …", end="\r")

    writer.release()
    print(f"\n  ✅  Test video saved: {out_path}  ({W}×{H} @ {fps}fps, {seconds}s)")


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic UAV test video")
    ap.add_argument("--out",     default="test_input.mp4")
    ap.add_argument("--drones",  type=int, default=3)
    ap.add_argument("--seconds", type=int, default=10)
    ap.add_argument("--fps",     type=int, default=25)
    ap.add_argument("--width",   type=int, default=1280)
    ap.add_argument("--height",  type=int, default=720)
    args = ap.parse_args()
    generate(args.out, args.width, args.height, args.fps, args.seconds, args.drones)


if __name__ == "__main__":
    main()
