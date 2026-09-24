import json
from pathlib import Path


# ============================================================
# WHISPER BACKEND
# ============================================================

try:
    from faster_whisper import WhisperModel

    _BACKEND = "faster"

except ImportError:
    WhisperModel = None
    _BACKEND = None


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = Path("data/raw/test_video.mp4")

OUTPUT_PATH = Path(
    "data/processed/test_video/transcript.json"
)

MODEL_NAME = "base"


# ============================================================
# TRANSCRIBE VIDEO
# ============================================================

def transcribe_video(video_path: Path):

    if _BACKEND != "faster":
        raise ImportError(
            "faster-whisper is not installed.\n"
            "Install it using:\n"
            "pip install faster-whisper"
        )

    print("\nLoading Whisper model...")

    try:
        model = WhisperModel(
            MODEL_NAME,
            device="cpu",
            compute_type="int8"
        )

    except Exception as e:

        print("\nERROR: Could not load Whisper model.")
        print("Reason:", e)

        return {
            "segments": []
        }

    print("Transcribing video using faster-whisper...\n")

    try:

        segments_generator, info = model.transcribe(
            str(video_path),
            beam_size=5,
            vad_filter=True
        )

    except Exception as e:

        print(
            "\nWarning: faster-whisper failed to decode audio:",
            e
        )

        print(
            "Skipping transcription for this file "
            "and returning empty segments."
        )

        return {
            "segments": []
        }

    # --------------------------------------------------------
    # Convert Whisper segments to normal dictionaries
    # --------------------------------------------------------

    segments = []

    try:

        for segment in segments_generator:

            # faster-whisper returns Segment objects
            start = float(segment.start)
            end = float(segment.end)
            text = str(segment.text).strip()

            if not text:
                continue

            segments.append(
                {
                    "start": start,
                    "end": end,
                    "text": text
                }
            )

    except Exception as e:

        print(
            "\nWarning: Error while reading transcription segments:",
            e
        )

        print(
            "Returning the segments collected so far."
        )

    return {
        "segments": segments
    }


# ============================================================
# SAVE TRANSCRIPT
# ============================================================

def save_transcript(result):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    segments = result.get(
        "segments",
        []
    )

    cleaned_segments = []

    for segment in segments:

        cleaned_segments.append(
            {
                "start": float(
                    segment.get("start", 0)
                ),
                "end": float(
                    segment.get("end", 0)
                ),
                "text": str(
                    segment.get("text", "")
                ).strip()
            }
        )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            cleaned_segments,
            f,
            indent=4,
            ensure_ascii=False
        )

    return cleaned_segments


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("VIDEO RAG - AUDIO TRANSCRIPTION")
    print("=" * 60)

    # --------------------------------------------------------
    # Check video
    # --------------------------------------------------------

    if not VIDEO_PATH.exists():

        raise FileNotFoundError(
            f"\nVideo not found:\n{VIDEO_PATH}\n\n"
            f"Make sure your video exists at:\n"
            f"data/raw/test_video.mp4"
        )

    print(
        f"\nInput video:\n{VIDEO_PATH}"
    )

    # --------------------------------------------------------
    # Transcribe
    # --------------------------------------------------------

    result = transcribe_video(
        VIDEO_PATH
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    segments = save_transcript(
        result
    )

    # --------------------------------------------------------
    # Output information
    # --------------------------------------------------------

    print(
        "\nTranscript saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        f"\nSegments detected: {len(segments)}"
    )

    # --------------------------------------------------------
    # Preview
    # --------------------------------------------------------

    if segments:

        print("\nTranscript preview:")

        for segment in segments[:5]:

            print(
                f"[{segment['start']:.2f}s - "
                f"{segment['end']:.2f}s] "
                f"{segment['text']}"
            )

    else:

        print(
            "\nNo transcript segments were generated."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()