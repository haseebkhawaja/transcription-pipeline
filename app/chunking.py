"""
Long-audio handling.

Why chunk at all, given that Whisper-family models can technically accept
arbitrarily long audio?
  - Memory: decoding + running inference on a multi-hour file at once can
    blow up memory on modest hardware.
  - Latency / timeouts: a synchronous HTTP request shouldn't block for the
    time it takes to transcribe a 3-hour file. Chunking lets us report
    progress and, in the API layer, move long jobs to a background task.
  - Fault isolation: if one chunk fails (e.g. a corrupt byte range), we
    lose one chunk's worth of transcript instead of the entire file.
  - Parallelization: independent chunks can be transcribed concurrently
    (multiple workers / GPU batching) instead of one long serial pass.

Strategy:
  - Files under CHUNK_THRESHOLD_SEC are transcribed as a single chunk --
    no need to add complexity for short files.
  - Longer files are split into fixed-length windows of CHUNK_LENGTH_SEC,
    with OVERLAP_SEC of overlap between consecutive chunks so that words
    spoken right at a cut point aren't lost or truncated.
  - We cut on fixed time windows rather than trying to detect silence for
    simplicity and predictability; the overlap + merge-time dedup step
    (see transcriber.merge_chunk_segments) compensates for boundary artifacts.
    A silence-aware splitter (e.g. via ffmpeg's silencedetect filter) is a
    natural upgrade if word-boundary cuts turn out to be a real problem in
    production -- noted in the README as a future improvement.
"""
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List

from app.audio_utils import AudioProcessingError

logger = logging.getLogger(__name__)

CHUNK_THRESHOLD_SEC = 15 * 60   # only bother chunking beyond 15 minutes
CHUNK_LENGTH_SEC = 10 * 60      # 10-minute chunks
OVERLAP_SEC = 2.0               # 2s overlap between consecutive chunks


@dataclass
class AudioChunk:
    path: str
    start_offset: float   # where this chunk starts, in seconds, in the ORIGINAL file
    end_offset: float


def plan_chunks(duration_sec: float) -> List[tuple]:
    """
    Pure function (no I/O) that decides chunk boundaries given a duration.
    Split out from split_audio() so the chunk-planning logic can be unit
    tested without touching ffmpeg or the filesystem.

    Returns a list of (start, end) tuples in seconds, covering the whole
    file, with OVERLAP_SEC overlap between consecutive windows.
    """
    if duration_sec <= CHUNK_THRESHOLD_SEC:
        return [(0.0, duration_sec)]

    windows = []
    start = 0.0
    while start < duration_sec:
        end = min(start + CHUNK_LENGTH_SEC, duration_sec)
        windows.append((start, end))
        if end >= duration_sec:
            break
        start = end - OVERLAP_SEC  # step forward, backing off by the overlap
    return windows


def split_audio(normalized_path: str, duration_sec: float, out_dir: str) -> List[AudioChunk]:
    """
    Physically cut `normalized_path` into chunk files according to
    plan_chunks(). Assumes the input is already normalized (16kHz mono
    WAV) so each chunk is cheap to cut with -c copy... in practice we
    re-encode (no -c copy) because WAV PCM cutting on arbitrary timestamps
    is already sample-accurate and cheap, avoiding any codec-copy edge
    cases.
    """
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    windows = plan_chunks(duration_sec)
    logger.info("Cutting %.2fs of audio into %d window(s).", duration_sec, len(windows))
    chunks = []
    for i, (start, end) in enumerate(windows):
        chunk_path = out_dir_path / f"chunk_{i:04d}.wav"
        logger.debug("Cutting chunk %d [%.1fs - %.1fs] -> %s", i, start, end, chunk_path)
        cmd = [
            "ffmpeg", "-y",
            "-i", str(normalized_path),
            "-ss", f"{start:.3f}",
            "-to", f"{end:.3f}",
            str(chunk_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise AudioProcessingError(
                f"Failed to cut chunk {i} [{start:.1f}s - {end:.1f}s]: "
                f"{result.stderr.strip()}"
            )
        chunks.append(AudioChunk(path=str(chunk_path), start_offset=start, end_offset=end))

    return chunks
