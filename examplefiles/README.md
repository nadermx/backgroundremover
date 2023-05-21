# Examples

This directory contains examples created using this library.

## Combine Images Using ffmpeg

Using the following command (assuming ffmpeg is installed) to combined `img1.jpg` and `img2.jpg` into a single side-by-side.

```bash
ffmpeg -i img1.jpg -i img2.jpg -filter_complex "[0:v]scale=iw/2:-1,pad=2*iw[left];[1:v]scale=iw/2:-1[right];[left][right]overlay=w" output.jpg
```
