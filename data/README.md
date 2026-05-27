# Data folder

Put your video files here. They are **not** committed (see `.gitignore`).

## HD-EPIC

Register and download from <https://hd-epic.github.io/>. Videos are organised
per participant (`P01`, `P02`, …) and per session. The full kitchen videos
are typically 30+ minutes long, so the baseline pipeline supports
clip-window mode (`--start-sec`, `--duration-sec`) to process a short
window at a time.

## Test clips

For quick iteration during development, save a short clip (5–30 s) here as
`sample.mp4`. Anything in `data/*.mp4` is git-ignored.

## Expected layout

```
data/
├── README.md
├── sample.mp4               # short clip for quick tests (not committed)
└── hd-epic/                 # full HD-EPIC structure (not committed)
    ├── P01/
    │   └── session-01.mp4
    └── P02/
        └── session-01.mp4
```
