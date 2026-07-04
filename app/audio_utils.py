"""
Audio format handling.

Design decision: instead of trying to support every input format inside the
transcription engine, we normalize EVERYTHING to a single canonical format
up front (16kHz, mono, 16-bit PCM WAV) using ffmpeg. This is the format
Whisper-family models expect internally anyway, so normalizing early:

  1. Lets the rest of the pipeline (chunking, transcription) be format-agnostic.
  2. Gives us one place to validate/reject corrupt or unsupported files.
  3. Avoids surprises from stereo files, unusual sample rates, or containers
     like m4a/ogg/webm that the STT library may not accept directly.

We use ffmpeg/ffprobe directly via subprocess (rather than pydub) so this
module has zero Python dependencies beyond the stdlib -- ffmpeg is the only
external binary required, and it's practically universal.
"""
import json
import logging
import shutil
import subprocess

from app.models import AudioInfo

logger = logging.getLogger(__name__)


class AudioProcessingError(Exception):
    """Raised when a file can't be probed or converted (corrupt/unsupported)."""


def _run(cmd: list) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def check_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise AudioProcessingError(
            "ffmpeg/ffprobe not found on PATH. Install ffmpeg to run this service."
        )


def probe_audio(path: str) -> AudioInfo:
    """
    Inspect a file with ffprobe to confirm it's readable audio and pull out
    metadata. This is our validation step: if ffprobe can't parse it,
    we reject the file with a clear error instead of letting it fail deep
    inside the STT model with a confusing stack trace.
    """
    check_ffmpeg_available()

    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration,format_name",
        "-show_entries", "stream=sample_rate,channels,codec_name,codec_type",
        "-of", "json",
        str(path),
    ]
    result = _run(cmd)
    if result.returncode != 0:
        raise AudioProcessingError(
            f"Could not read '{path}' as audio/video. "
            f"File may be corrupt or an unsupported format.\n"
            f"ffprobe error: {result.stderr.strip()}"
        )

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise AudioProcessingError(f"Unexpected ffprobe output for '{path}': {e}")

    audio_streams = [
        s for s in data.get("streams", []) if s.get("codec_type") == "audio"
    ]
    if not audio_streams:
        raise AudioProcessingError(
            f"'{path}' does not contain an audio stream (is this a video-only "
            f"or non-media file?)."
        )

    stream = audio_streams[0]
    fmt = data.get("format", {})

    if "duration" not in fmt:
        raise AudioProcessingError(
            f"Could not determine duration for '{path}' (possibly truncated file)."
        )

    return AudioInfo(
        path=str(path),
        duration_sec=float(fmt["duration"]),
        sample_rate=int(stream.get("sample_rate", 0)),
        channels=int(stream.get("channels", 0)),
        codec=stream.get("codec_name", "unknown"),
        format_name=fmt.get("format_name", "unknown"),
    )


def normalize_audio(input_path: str, output_path: str) -> AudioInfo:
    """
    Convert any input (mp3, m4a, ogg, flac, wav-with-odd-sample-rate,
    stereo, etc.) into a canonical 16kHz mono 16-bit PCM WAV file.

    Notes on format handling:
      - We deliberately ignore the file *extension* for routing -- ffmpeg
        sniffs the actual container/codec from the file bytes, so a
        mislabeled extension (e.g. an .mp3 that's actually a .wav) still
        works, and a genuinely bogus file is caught by probe_audio() above.
      - -ac 1 downmixes any channel count to mono (STT accuracy doesn't
        benefit from stereo, and it halves the data volume).
      - -ar 16000 resamples to 16kHz, the standard STT sample rate.
      - -sample_fmt s16 gives us consistent 16-bit PCM regardless of the
        source bit depth.
    """
    check_ffmpeg_available()

    # Validate first so we fail fast with a clear message.
    probe_audio(input_path)

    logger.info("Normalizing '%s' -> '%s' (16kHz mono s16 WAV).", input_path, output_path)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        "-vn",  # drop any video/album-art stream some mp3/m4a files embed
        str(output_path),
    ]
    result = _run(cmd)
    if result.returncode != 0:
        raise AudioProcessingError(
            f"ffmpeg failed to normalize '{input_path}':\n{result.stderr.strip()}"
        )

    return probe_audio(output_path)
