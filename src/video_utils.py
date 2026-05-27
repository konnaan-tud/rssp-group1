"""Helpers for extracting short clips from longer cooking videos.

HD-EPIC videos are typically 30+ minutes long. We almost never want to feed
the entire video to a VLM. This module provides:

  - `extract_clip(video_path, start_sec, duration_sec, out_path)`
    Cuts a clip with ffmpeg-via-opencv (no audio).

Keep external dependencies light: opencv-python only.
"""

from __future__ import annotations
from pathlib import Path

import cv2


def extract_clip(
    video_path: str | Path,
    start_sec: float,
    duration_sec: float,
    out_path: str | Path,
    target_fps: float | None = None,
) -> Path:
    """Cut a clip from `video_path` between `start_sec` and start+duration.

    Parameters
    ----------
    video_path : path to the source video
    start_sec  : start time in seconds
    duration_sec : length of the clip in seconds
    out_path   : where to save the resulting clip (mp4)
    target_fps : if given, re-encode to this fps (useful to reduce token cost
                 to the VLM). If None, keep the source fps.

    Returns
    -------
    Path to the written clip.
    """
    video_path = Path(video_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = target_fps if target_fps else src_fps

    start_frame = int(round(start_sec * src_fps))
    end_frame = int(round((start_sec + duration_sec) * src_fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (width, height))

    frame_step = max(1, int(round(src_fps / out_fps))) if target_fps else 1
    current = start_frame
    while current < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if (current - start_frame) % frame_step == 0:
            writer.write(frame)
        current += 1

    cap.release()
    writer.release()
    return out_path


def get_video_duration(video_path: str | Path) -> float:
    """Return duration of `video_path` in seconds."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0:
        raise ValueError(f"Invalid fps reported for {video_path}: {fps}")
    return frames / fps
