# Supplementary videos

The project page provides an interactive browser for the real-robot videos:

<https://renshaojie233.github.io/StreamingWAM/>

The videos are distributed as assets of the
[`supplementary-v1`](https://github.com/renshaojie233/StreamingWAM/releases/tag/supplementary-v1)
release rather than Git objects. This keeps normal repository clones small.

The release contains:

- 7 complete real-robot task demonstrations;
- 48 comparison videos: 2 tasks × 3 policies × 2 trials × 4 camera views;
- 4 views of an additional human-assisted trial;
- one ZIP archive containing all 59 MP4 files and their SHA-256 manifest.

All MP4 assets use H.264 video with `yuv420p` pixel format for browser
compatibility. See [`video-manifest.json`](video-manifest.json) for filenames,
file sizes, original relative paths, and checksums.
