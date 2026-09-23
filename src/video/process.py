import cv2
import json
from pathlib import Path


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

VIDEO_PATH = Path("data/raw/test_video.mp4")

OUTPUT_DIR = Path(
    "data/processed/test_video"
)

FRAMES_DIR = OUTPUT_DIR / "frames"

FRAME_INTERVAL = 5       # seconds
CHUNK_DURATION = 30      # seconds


# --------------------------------------------------
# FORMAT TIME
# --------------------------------------------------

def format_timestamp(seconds):

    hours = int(seconds // 3600)

    minutes = int(
        (seconds % 3600) // 60
    )

    secs = seconds % 60

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:05.2f}"
    )


# --------------------------------------------------
# EXTRACT FRAMES
# --------------------------------------------------

def extract_frames():

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: "
            f"{VIDEO_PATH}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    duration = frame_count / fps

    # Create frames directory
    FRAMES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    frames = []

    current_time = 0.0

    while current_time < duration:

        # Convert seconds → frame number
        frame_number = int(
            current_time * fps
        )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number
        )

        success, frame = cap.read()

        if not success:

            print(
                f"Could not read frame "
                f"at {current_time:.2f}s"
            )

            current_time += FRAME_INTERVAL

            continue

        filename = (
            f"frame_"
            f"{int(current_time):06d}.jpg"
        )

        output_path = (
            FRAMES_DIR / filename
        )

        cv2.imwrite(
            str(output_path),
            frame
        )

        frames.append(
            {
                "timestamp": current_time,
                "timestamp_formatted":
                    format_timestamp(
                        current_time
                    ),
                "frame_number":
                    frame_number,
                "file":
                    str(output_path)
            }
        )

        print(
            f"Extracted: "
            f"{format_timestamp(current_time)}"
        )

        current_time += FRAME_INTERVAL

    cap.release()

    return frames


# --------------------------------------------------
# CREATE VIDEO CHUNKS
# --------------------------------------------------

def create_chunks(
    frames,
    duration
):

    chunks = []

    chunk_start = 0.0

    chunk_id = 0

    while chunk_start < duration:

        chunk_end = min(
            chunk_start + CHUNK_DURATION,
            duration
        )

        # Find frames belonging
        # to this chunk

        chunk_frames = [
            frame
            for frame in frames
            if (
                frame["timestamp"]
                >= chunk_start
                and
                frame["timestamp"]
                < chunk_end
            )
        ]

        chunk = {

            "chunk_id": chunk_id,

            "start": chunk_start,

            "end": chunk_end,

            "start_formatted":
                format_timestamp(
                    chunk_start
                ),

            "end_formatted":
                format_timestamp(
                    chunk_end
                ),

            "frames":
                chunk_frames,

            "transcript": None,

            "ocr": None,

            "visual_description": None
        }

        chunks.append(chunk)

        chunk_id += 1

        chunk_start += CHUNK_DURATION

    return chunks


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    if not VIDEO_PATH.exists():

        raise FileNotFoundError(
            f"Video not found: "
            f"{VIDEO_PATH}"
        )

    print("\n")
    print("=" * 60)
    print("VIDEO PROCESSING")
    print("=" * 60)

    # Open video
    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    cap.release()

    duration = frame_count / fps

    print(
        f"Duration: "
        f"{format_timestamp(duration)}"
    )

    print(
        f"Sampling: "
        f"1 frame every "
        f"{FRAME_INTERVAL} seconds"
    )

    print(
        f"Chunk size: "
        f"{CHUNK_DURATION} seconds"
    )

    print("=" * 60)

    # ------------------------------------------
    # Extract frames
    # ------------------------------------------

    print("\nExtracting frames...\n")

    frames = extract_frames()

    print(
        f"\nTotal frames extracted: "
        f"{len(frames)}"
    )

    # ------------------------------------------
    # Create chunks
    # ------------------------------------------

    print("\nCreating chunks...\n")

    chunks = create_chunks(
        frames,
        duration
    )

    print(
        f"Total chunks created: "
        f"{len(chunks)}"
    )

    # ------------------------------------------
    # Save chunks
    # ------------------------------------------

    output_file = (
        OUTPUT_DIR / "chunks.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            chunks,
            f,
            indent=4
        )

    print(
        f"\nChunks saved to:"
        f"\n{output_file}"
    )

    print("\nDone!")


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    main()