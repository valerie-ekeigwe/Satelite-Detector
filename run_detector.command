#!/bin/bash
# Double-click me to run the detector (macOS)

# Go to this script's folder
cd "$(dirname "$0")"

# Activate venv (create+install if missing)
if [[ ! -d ".venv" ]]; then
  /usr/bin/python3 -m venv .venv || exit 1
  source .venv/bin/activate
  python -m pip install --upgrade pip
  pip install numpy opencv-python
else
  source .venv/bin/activate
fi

# Choose video: drag & drop a file onto this icon, or default to starlink_test.mp4
VIDEO="${1:-starlink_test.mp4}"

# Run the detector
exec python satellite_detector.py \
  --video "$VIDEO" \
  --show \
  --save-csv tracks.csv \
  --live-count

