# Riesz Pyramids for Fast Phase-Based Video Magnifiction

Python rewrite of portion of code from supplemental material of

> Riesz Pyramids for Fast Phase-Based Video Magnification
> Neal Wadhwa, Michael Rubinstein, Fredo Durand and William T. Freeman
> Computational Photography (ICCP), 2014 IEEE International Conference on

I am not affiliated with the authors. I just wanted to rewrite the code in Python
to better understand it, and so that I didn't need some obscure version of MatLab
(which I don't have access to anymore).

Find the supplemental material this is based off [here](https://people.csail.mit.edu/nwadhwa/riesz-pyramid/)!

A [random video](https://youtu.be/jNQXAC9IVRw?si=YB3ypp11UWI1qqMs) is included for your convenience.

## Things to play around with

- `ALPHA`: Magnification factor
- `f_lo`, `f_hi`: low and high frequency cutoff for bandpass filter.
- `fft_level` option in `fft_amplify_pyr_frames`. This controls the spatial octaves that get amplified.
  This takes an iterable (e.g. `[-4,-3]`). 0 is lowest (blurriest) layer, -1 is highest (sharpest) layer.
- **Experts only:** `reconstruct_amplified_pyr` function. I'm just setting the output `pyr_frames` to be the amplified frames `amplified_pyr_frames`. However, you can add the amplification to the output, which is the original intent.

## Requirements

- FFmpeg
- Python3 + following dependencies:
  - numpy
  - scipy
  - pillow
  - tqdm

## [LICENSE](LICENSE.pdf)

**TLDR:** This software is for non-commercial research and/or academic testing purposes only!
