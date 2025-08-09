import argparse
import csv
from collections import deque
from dataclasses import dataclass
from math import hypot
from typing import List, Tuple
import cv2
import numpy as np
def to_gray(frame: np.ndarray, sigma: float) -> np.ndarray:
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    f = frame.astype(np.float32)
    if sigma > 0:
        k = max(1, int(2 * sigma + 1))
        f = cv2.blur(f, (k, 1))
        f = cv2.blur(f, (1, k))
    return np.clip(f, 0, 255).astype(np.uint8)
class RunningPercentile:
    def __init__(self, window: int = 31, q: float = 50):
        self.window = int(window)
        self.q = q
        self.buf: deque[np.ndarray] = deque(maxlen=self.window)
    def update(self, gray: np.ndarray):
        self.buf.append(gray.copy())
    def ready(self) -> bool:
        return len(self.buf) >= max(5, self.window // 2)
    def background(self) -> np.ndarray:
        arr = np.stack(self.buf, axis=0)
        return np.percentile(arr, self.q, axis=0).astype(np.uint8)
def robust_mask(diff: np.ndarray, k: float, bright_clip: int) -> np.ndarray:
    valid = diff < bright_clip
    vals = diff[valid]
    if vals.size < 100:
        thr = int(np.mean(diff) + k * (np.std(diff) + 1e-6))
        return (diff > max(10, thr)).astype(np.uint8)
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) + 1e-6
    z = (diff - med) / (1.4826 * mad)
    return (z > k).astype(np.uint8)
def morph_clean(bin_img: np.ndarray, dilate_iters: int, erode_iters: int) -> np.ndarray:
    k = np.ones((3, 3), np.uint8)
    out = bin_img
    if erode_iters > 0:
        out = cv2.erode(out, k, iterations=erode_iters)
    if dilate_iters > 0:
        out = cv2.dilate(out, k, iterations=dilate_iters)
    return out
def find_blobs(binary: np.ndarray, min_area: int, max_area: int):
    num, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if min_area <= area <= max_area:
            cx, cy = centroids[i]
            out.append((float(cx), float(cy), int(area), (x, y, w, h)))
    return out
@dataclass
class Track:
    id: int
    points: List[Tuple[int, float, float]]
    misses: int = 0
    def length(self) -> int:
        return len(self.points)
    def last(self):
        return self.points[-1]
    def predict(self):
        if self.length() < 2:
            _, x, y = self.last()
            return x, y
        (_, x2, y2), (_, x1, y1) = self.points[-1], self.points[-2]
        
        return x2 + (x2 - x1), y2 + (y2 - y1)
    def add(self, fr: int, x: float, y: float):
        self.points.append((fr, x, y))
        self.misses = 0
class Tracker:
    def __init__(self, link_dist: float, max_miss: int):
        self._next_id = 1
        self.link_dist = float(link_dist)
        self.max_miss = int(max_miss)
        self.tracks: List[Track] = []
    def step(self, fr: int, detections):
        used = set()
        for t in self.tracks:
            tx, ty = t.predict()
            best_j = None
            best_d = 1e9
            for j, (cx, cy, _, _) in enumerate(detections):
                if j in used:
                    continue
                d = hypot(cx - tx, cy - ty)
                if d < best_d:
                    best_d, best_j = d, j
            if best_j is not None and best_d <= self.link_dist:
                cx, cy, _, _ = detections[best_j]
                t.add(fr, cx, cy)
                used.add(best_j)
            else:
                t.misses += 1
        for j, (cx, cy, _, _) in enumerate(detections):
            if j not in used:
                self.tracks.append(Track(id=self._next_id, points=[(fr, cx, cy)]))
                self._next_id += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_miss]
def satellite_like(track: Track, fps: float, min_len: int) -> bool:
    if track.length() < min_len:
        return False
    v = []
    for k in range(1, track.length()):
        _, x2, y2 = track.points[k]
        _, x1, y1 = track.points[k - 1]
        v.append(hypot(x2 - x1, y2 - y1))
    if len(v) < 3:
        return False
    v = np.asarray(v, dtype=float) * fps  
    mean_v = float(np.mean(v))
    if mean_v < 5 or mean_v > 500:
        return False
    
    if (np.std(v) / (mean_v + 1e-6)) > 0.4:
        return False
    return True
def run(args):
    src = 0 if str(args.video).strip() == "0" else args.video
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit("Cannot open video/camera.")
    fps = (cap.get(cv2.CAP_PROP_FPS) or 30.0) * args.fps_rescale
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    writer = cv2.VideoWriter(
        args.out,
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, fps),
        (W, H),
    )
    bg = RunningPercentile(window=args.bg_window, q=50)
    tracker = Tracker(link_dist=args.link_dist, max_miss=args.max_miss)
    confirmed_ids = set()
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = to_gray(frame, args.smoothing_sigma)
        bg.update(gray)
        if not bg.ready():
            vis = frame
        else:
            diff = cv2.absdiff(gray, bg.background())
            mask = robust_mask(diff, k=args.thresh_k, bright_clip=args.bright_clip)
            mask = morph_clean(mask, dilate_iters=args.dilate_iters, erode_iters=args.erode_iters)
            blobs = find_blobs(mask, min_area=args.min_area, max_area=args.max_area)
            tracker.step(frame_idx, blobs)
            
            for t in tracker.tracks:
                if t.id not in confirmed_ids and satellite_like(t, fps=fps, min_len=args.min_track):
                    confirmed_ids.add(t.id)
                    if args.live_count:
                        print(f"[+] New satellite ID {t.id} (total: {len(confirmed_ids)})")
            vis = frame.copy()
            for (cx, cy, _, _) in blobs:
                cv2.circle(vis, (int(cx), int(cy)), 3, (0, 255, 255), 1)
            for t in tracker.tracks:
                color = (0, 220, 0) if t.id in confirmed_ids else (200, 200, 200)
                tail = t.points[-args.draw_trail:]
                for k in range(1, len(tail)):
                    _, x2, y2 = tail[k]
                    _, x1, y1 = tail[k - 1]
                    cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
                _, tx, ty = t.last()
                cv2.putText(vis, f"ID {t.id}", (int(tx) + 5, int(ty) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        hud = f"fps:{fps:.1f} tracks:{len(tracker.tracks)} frame:{frame_idx} sats:{len(confirmed_ids)}"
        cv2.putText(vis, hud, (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(vis)
        if args.show:
            cv2.imshow("satellite-detector", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        frame_idx += 1
    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()
    if args.save_csv:
        with open(args.save_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["track_id", "frame", "x", "y"])
            for t in tracker.tracks:
                for fr, x, y in t.points:
                    w.writerow([t.id, fr, x, y])
    if args.live_count:
        print(f"[summary] confirmed satellites: {len(confirmed_ids)}")
def parse_args():
    p = argparse.ArgumentParser(description="Detect satellites in night-sky video.")
    p.add_argument("--video", default="night_sky.mp4")
    p.add_argument("--out", default="annotated.mp4")
    p.add_argument("--save-csv", default="")
    p.add_argument("--show", action="store_true")
    p.add_argument("--live-count", action="store_true")
    p.add_argument("--bg-window", type=int, default=31)
    p.add_argument("--thresh-k", type=float, default=3.5)
    p.add_argument("--min-area", type=int, default=2)
    p.add_argument("--max-area", type=int, default=30)
    p.add_argument("--link-dist", type=float, default=12)
    p.add_argument("--min-track", type=int, default=8)
    p.add_argument("--max-miss", type=int, default=2)
    p.add_argument("--smoothing-sigma", type=float, default=1.0)
    p.add_argument("--dilate-iters", type=int, default=1)
    p.add_argument("--erode-iters", type=int, default=1)
    p.add_argument("--bright-clip", type=int, default=240)
    p.add_argument("--draw-trail", type=int, default=30)
    p.add_argument("--fps-rescale", type=float, default=1.0)
    return p.parse_args()
if __name__ == "__main__":
    run(parse_args())
