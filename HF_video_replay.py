import argparse
import os
import re
import sys

import cv2
import imageio
import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files

WINDOW_NAME = "Unitree G1 Multi-View Replay"
DEFAULT_REPO_ID = "unitreerobotics/G1_Dex3_PickBottle_Dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Selectively download and replay first MP4 per detected camera folder."
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repo id (default: {DEFAULT_REPO_ID})",
    )
    return parser.parse_args()


def detect_camera_key(path: str) -> str:
    # Prefer explicit camera token (cam_*) when present in path.
    token_match = re.search(r"(cam_[a-zA-Z0-9_]+)", path)
    if token_match:
        return token_match.group(1)

    # Fallback to the immediate folder under videos/.
    parts = path.split("/")
    try:
        video_idx = parts.index("videos")
        if video_idx + 1 < len(parts):
            return parts[video_idx + 1]
    except ValueError:
        pass
    return "unknown_camera"


def find_first_mp4_per_camera(repo_id: str) -> dict[str, str]:
    print(f"1) Inspecting repository files: {repo_id}")
    all_files = list_repo_files(repo_id=repo_id, repo_type="dataset")

    mp4_files = [
        f for f in all_files if f.startswith("videos/") and f.lower().endswith(".mp4")
    ]
    if not mp4_files:
        return {}

    camera_to_files: dict[str, list[str]] = {}
    for file_path in mp4_files:
        camera_key = detect_camera_key(file_path)
        camera_to_files.setdefault(camera_key, []).append(file_path)

    first_mp4_per_camera: dict[str, str] = {}
    for camera_key, files in camera_to_files.items():
        first_mp4_per_camera[camera_key] = sorted(files)[0]

    return dict(sorted(first_mp4_per_camera.items()))


def download_selected_videos(repo_id: str, selected_files: dict[str, str]) -> dict[str, str]:
    local_paths: dict[str, str] = {}
    print("2) Downloading only one MP4 per detected camera...")
    for camera_key, remote_file in selected_files.items():
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=remote_file,
        )
        local_paths[camera_key] = os.path.abspath(local_path)
        print(f"   [OK] {camera_key}: {remote_file}")
    return local_paths


def print_downloaded_paths(local_paths: dict[str, str]) -> None:
    print("\n3) Downloaded/Cached local file paths (absolute):")
    for camera_key, abs_path in local_paths.items():
        print(f"   {camera_key}: {abs_path}")


def build_grid(frames: list[np.ndarray], labels: list[str]) -> np.ndarray:
    if not frames:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    h, w = frames[0].shape[:2]
    resized = [cv2.resize(frame, (w, h)) for frame in frames]

    annotated: list[np.ndarray] = []
    for frame, label in zip(resized, labels):
        canvas = frame.copy()
        cv2.putText(
            canvas,
            label,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        annotated.append(canvas)

    if len(annotated) % 2 == 1:
        annotated.append(np.zeros_like(annotated[0]))

    rows = []
    for i in range(0, len(annotated), 2):
        rows.append(np.hstack((annotated[i], annotated[i + 1])))
    return np.vstack(rows)


def replay_videos(local_paths: dict[str, str]) -> None:
    readers: dict[str, object] = {}
    for camera_key, path in local_paths.items():
        try:
            readers[camera_key] = imageio.get_reader(path)
        except Exception as exc:
            print(f"[WARN] Could not open reader for {camera_key}: {exc}")

    if not readers:
        print("No readable videos found. Exiting.")
        return

    camera_order = sorted(readers.keys())
    print("\n4) Starting multi-view replay. Press 'q' or ESC to quit.")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            current_frames = []
            for camera_key in camera_order:
                try:
                    frame = readers[camera_key].get_next_data()
                except (EOFError, IndexError, StopIteration):
                    print(f"\nEnd of video reached: {camera_key}")
                    raise StopIteration
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                current_frames.append(frame_bgr)

            grid = build_grid(current_frames, camera_order)
            display = cv2.resize(grid, (0, 0), fx=0.8, fy=0.8)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(30) & 0xFF
            try:
                visible = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
            except cv2.error:
                visible = -1

            if key == ord("q") or key == 27 or visible < 1:
                print("\nReplay stopped by user.")
                break

    except StopIteration:
        print("All selected camera videos finished.")
    finally:
        for reader in readers.values():
            reader.close()
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()

    try:
        selected = find_first_mp4_per_camera(args.repo_id)
    except Exception as exc:
        print(f"Failed to inspect repository files: {exc}")
        sys.exit(1)

    if not selected:
        print("No MP4 files found under videos/ in the target dataset repository.")
        sys.exit(1)

    print("Detected camera -> first MP4:")
    for cam, remote_file in selected.items():
        print(f"   {cam}: {remote_file}")

    try:
        local_paths = download_selected_videos(args.repo_id, selected)
    except Exception as exc:
        print(f"Failed while downloading selected videos: {exc}")
        sys.exit(1)

    print_downloaded_paths(local_paths)
    replay_videos(local_paths)


if __name__ == "__main__":
    main()