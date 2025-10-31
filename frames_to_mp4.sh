#!/usr/bin/bash
set -euo pipefail

INPUT_FN="input.mp4"
OUTPUT_FN="out.mp4"

## To run with just the video (no audio):
##   Don't forget to set the correct framerate!
# ffmpeg -r 15.0003 -i frames_out/%03d.png -c:v libx264 \
#   -pix_fmt yuv420p $OUTPUT_FN -y

## To map the audio from the original input file to the output:
ffmpeg -r 15.0003 -i frames_out/%03d.png -i $INPUT_FN -c:v libx264 \
  -pix_fmt yuv420p -map 0:v:0 -map 1:a:0 $OUTPUT_FN -y
