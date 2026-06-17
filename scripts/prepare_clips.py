"""
scripts/prepare_clips.py
------------------------
One-time preparation for the carbonara real-video session.

Takes the raw videos in data/videos/ and produces an ordered set of 5-second
clips in data/clips/. Two transforms:

  1. Split data/videos/drain_pasta+pasta_into_skillet.mp4 at the 13-second
     mark — first part is "draining pasta over the sink", second is
     "transferring pasta into the skillet".
  2. Trim every video to its first 5 seconds.

Output is numbered so the orchestrator can iterate in order.

Requires ffmpeg on PATH. Install with: brew install ffmpeg

Run from the repo root:
    python scripts/prepare_clips.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = REPO_ROOT / "data" / "videos" / "carbonara"
OUTPUT_DIR = REPO_ROOT / "data" / "clips"

# Name of the merged video that needs splitting.
MERGED_VIDEO = "drain_pasta+pasta_into_skillet.mp4"
SPLIT_POINT_SEC = 13.0
CLIP_DURATION_SEC = 5.0

# Carbonara session in recipe order. Each entry is
#   (output_name, source_filename, start_seconds)
# Use the special source __MERGED__ to mean "pull from the merged video".
ORDERED_CLIPS = [
    ("01_pour_water.mp4",            "pour_water.mp4",         0.0),
    ("02_crack_egg.mp4",             "crack_egg.mp4",          0.0),
    ("03_grating_pecorino.mp4",      "grating_pecorino.mp4",   0.0),
    ("04_chopping_pancetta.mp4",     "chopping_pancetta.mp4",  0.0),
    ("05_cooking_pancetta.mp4",      "cooking_pancetta.mp4",   0.0),
    ("06_boil_pasta.mp4",            "boil_pasta.mp4",         0.0),
    ("07_drain_pasta.mp4",           "__MERGED__",             0.0),
    ("08_pasta_into_skillet.mp4",    "__MERGED__",             SPLIT_POINT_SEC),
    ("09_stirring_carbonara.mp4",    "stirring_carbonara.mp4", 0.0),
]


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        print(
            "ERROR: ffmpeg not found on PATH.\n"
            "Install with: brew install ffmpeg",
            file=sys.stderr,
        )
        sys.exit(1)


def trim_clip(
    src: Path,
    dst: Path,
    start: float = 0.0,
    duration: float = CLIP_DURATION_SEC,
) -> None:
    """Extract `duration` seconds from `src` starting at `start`, re-encoded."""
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Two-pass argument order: -ss BEFORE -i for fast seek to keyframe, then
    # -t after -i for duration. Re-encode (libx264) so the cut is exact at
    # the boundary, not snapped to the previous keyframe.
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start}",
        "-i", str(src),
        "-t", f"{duration}",
        "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-loglevel", "error",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    check_ffmpeg()

    if not INPUT_DIR.exists():
        print(f"ERROR: input directory not found: {INPUT_DIR}", file=sys.stderr)
        sys.exit(1)

    merged_src = INPUT_DIR / MERGED_VIDEO
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Source:      {INPUT_DIR}")
    print(f"Destination: {OUTPUT_DIR}")
    print(f"Clip length: {CLIP_DURATION_SEC} s\n")

    missing: list[str] = []
    written: list[Path] = []

    for dst_name, src_name, start in ORDERED_CLIPS:
        dst = OUTPUT_DIR / dst_name

        if src_name == "__MERGED__":
            if not merged_src.exists():
                missing.append(MERGED_VIDEO)
                print(f"  [skip] {dst_name} — missing merged source")
                continue
            label = f"merged @ {start:.0f}s"
            src = merged_src
        else:
            src = INPUT_DIR / src_name
            if not src.exists():
                missing.append(src_name)
                print(f"  [skip] {dst_name} — missing source {src_name}")
                continue
            label = src_name

        print(f"  [trim] {label:<45} → {dst_name}")
        trim_clip(src, dst, start=start, duration=CLIP_DURATION_SEC)
        written.append(dst)

    print()
    if missing:
        print("Missing input(s):")
        for m in missing:
            print(f"  - {m}")
        print()

    print(f"Wrote {len(written)} clip(s) to {OUTPUT_DIR}/")
    if written:
        print("\nNext step:")
        print("  PYTORCH_ENABLE_MPS_FALLBACK=1 python orchestrator_v2_vlm.py")


if __name__ == "__main__":
    main()
