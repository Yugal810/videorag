import cv2
import json
import subprocess
from pathlib import Path


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

VIDEO_PATH = Path("data/raw/test_video.mp4")
OUTPUT_PATH = Path("data/processed/metadata.json")


# --------------------------------------------------
# HELPER: FORMAT DURATION
# --------------------------------------------------

def format_duration(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"


# --------------------------------------------------
# GET VIDEO INFORMATION USING OPENCV
# --------------------------------------------------

def inspect_video(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    cap.release()

    if fps <= 0:
        raise RuntimeError("Invalid FPS detected.")

    duration = frame_count / fps

    return {
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": duration
    }


# --------------------------------------------------
# CHECK WHETHER AUDIO EXISTS
# --------------------------------------------------

def has_audio(video_path):
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        str(video_path)
    ]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

    except FileNotFoundError:
        raise RuntimeError(
            "FFprobe was not found. "
            "Please install FFmpeg and make sure "
            "ffprobe is available in your PATH."
        )

    return bool(result.stdout.strip())


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    # Check that video exists
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}"
        )

    print("\nInspecting video...\n")

    # Video information
    video_info = inspect_video(VIDEO_PATH)

    # Audio information
    audio_available = has_audio(VIDEO_PATH)

    # Combine information
    metadata = {
        "file": VIDEO_PATH.name,
        "path": str(VIDEO_PATH),

        "has_video": True,
        "has_audio": audio_available,

        "duration_seconds":
            video_info["duration_seconds"],

        "duration":
            format_duration(
                video_info["duration_seconds"]
            ),

        "fps":
            video_info["fps"],

        "frame_count":
            video_info["frame_count"],

        "width":
            video_info["width"],

        "height":
            video_info["height"]
    }

    # Create output directory
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save JSON
    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    # Display results
    print("=" * 50)
    print("VIDEO INFORMATION")
    print("=" * 50)

    print(f"File        : {metadata['file']}")
    print(f"Duration    : {metadata['duration']}")
    print(f"Resolution  : "
          f"{metadata['width']} x "
          f"{metadata['height']}")

    print(f"FPS         : {metadata['fps']:.2f}")
    print(f"Frame count : {metadata['frame_count']}")

    print(
        f"Audio       : "
        f"{'YES' if metadata['has_audio'] else 'NO'}"
    )

    print("=" * 50)

    print(
        f"\nMetadata saved to:\n"
        f"{OUTPUT_PATH}\n"
    )


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    main()