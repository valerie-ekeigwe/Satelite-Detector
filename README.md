# Satellite Detector

A Python tool for spotting and tracking satellites in night sky videos. It works frame-by-frame: building a model of the static stars, detecting moving points, and tracking their motion until it’s clear they’re behaving like satellites.

---

## Project Background
This project is inspired by a simple satelite detector I had worked on for my Natural Foundations class, by combining solid image processing techniques with a simple, well-tuned tracker, it can reliably identify satellite passes in real footage, making it both a great technical demo and a useful observational tool. 

---

##  Features
- Adaptive background modeling to handle gradual sky changes
- Motion detection using a robust z-score method
- Morphology operations to remove noise and false positives
- Lightweight tracker with simple velocity-based prediction
- Heuristic filter to confirm smooth, satellite-like motion
- Live satellite counter overlay and console updates

---

## Requirements
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white)


```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate
pip install numpy opencv-python ```bash
```

## Quick Start
1. Place your night sky video in the project folder (e.g., `starlink_test.mp4`).
2. Run:
```bash
python satellite_detector.py --video starlink_test.mp4 --show --save-csv tracks.csv --live-count
```
## Tuned Commands
**Balanced detection**:
```bash
python satellite_detector.py --video starlink_test.mp4 --show --save-csv tracks.csv --live-count --thresh-k 3.2 --min-area 2 --max-area 20 --bg-window 60 --link-dist 14 --max-miss 3 --min-track 9 --smoothing-sigma 0.8


##How It Works
1. Convert frame to grayscale and apply light blur to reduce noise.
2. Build a running median background model for static stars.
3. Subtract background to reveal moving pixels.
4. Use robust z-score thresholding to detect movement.
5. Clean mask with erosion/dilation.
6. Detect blobs within a defined size range.
7. Track blobs using nearest-neighbor matching with velocity prediction:
   - **Velocity** (px/frame) = √((x₂ − x₁)² + (y₂ − y₁)²)
   - Convert to px/s using FPS
   - Predict next position: **x̂ₖ₊₁ = xₖ + (xₖ − xₖ₋₁)**, **ŷₖ₊₁ = yₖ + (yₖ − yₖ₋₁)**
8. Confirm as satellite if motion is smooth, within speed bounds, and lasts enough frames.
```
---

## Output
- **annotated.mp4** — detections and track trails (green = confirmed, grey = unconfirmed)
- **tracks.csv** — detection positions over time
- **HUD** — FPS, active tracks, frame count, confirmed satellites
