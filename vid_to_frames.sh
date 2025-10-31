#!/usr/bin/bash
set -euo pipefail

INPUT_FN="input.mp4"
## outputs frames to `frames_in`
mkdir -p frames_in
ffmpeg -i $INPUT_FN "frames_in/%03d.png"
