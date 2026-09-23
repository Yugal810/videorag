import json
from pathlib import Path

import whisper


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

VIDEO_PATH = Path(
    "data/raw/test_video.mp4"
)

OUTPUT_PATH = Path(
    "data/processed/test_video/transcript.json"
)

MODEL_NAME = "base"


# --------------------------------------------------
# TRANSCRIBE VIDEO
# --------------------------------------------------

def transcribe_video(video_path):

    print("\nLoading Whisper model...")

    model = whisper.load_model(
        MODEL_NAME
    )

    print("Transcribing video...\n")

    result = model.transcribe(
        str(video_path),
        verbose=False
    )

    return result


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    if not VIDEO_PATH.exists():

        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}"
        )

    result = transcribe_video(
        VIDEO_PATH
    )

    segments = []

    for segment in result["segments"]:

        segments.append(
            {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip()
            }
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            segments,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"\nTranscript saved to:"
        f"\n{OUTPUT_PATH}"
    )

    print(
        f"\nSegments detected: "
        f"{len(segments)}"
    )


if __name__ == "__main__":
    main()