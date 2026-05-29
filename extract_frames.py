import cv2
import os

# ── Settings ───────────────────────────────────────────────────────────────
CLIPS_DIR = "clips"
FRAMES_DIR = "frames"
FPS_SAMPLE = 2  # extract one frame every 0.5 seconds

# ── Extractor ──────────────────────────────────────────────────────────────
def extract_frames(clips_dir, frames_dir, fps_sample):
    clip_files = [f for f in os.listdir(clips_dir) if f.endswith(".mp4")]

    if not clip_files:
        print("No .mp4 files found in clips/")
        return

    for clip_file in clip_files:
        clip_path = os.path.join(clips_dir, clip_file)
        clip_name = os.path.splitext(clip_file)[0]
        output_dir = os.path.join(frames_dir, clip_name)
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(clip_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / native_fps if native_fps > 0 else 0

        # How many native frames to skip between each sample
        frame_interval = int(native_fps / fps_sample)

        print(f"\n{clip_file}")
        print(f"  Duration: {duration:.1f}s | Native FPS: {native_fps:.1f} | Interval: every {frame_interval} frames")

        saved = 0
        frame_idx = 0

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            filename = os.path.join(output_dir, f"frame_{frame_idx:05d}.jpg")
            cv2.imwrite(filename, frame)
            saved += 1
            frame_idx += frame_interval

        cap.release()
        print(f"  Saved {saved} frames → frames/{clip_name}/")

    print("\nDone.")

if __name__ == "__main__":
    extract_frames(CLIPS_DIR, FRAMES_DIR, FPS_SAMPLE)