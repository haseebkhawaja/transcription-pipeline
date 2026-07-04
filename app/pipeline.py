"""
Pipeline orchestrator: wires together format normalization, chunking,
per-chunk transcription, and merging into one call.

Kept separate from main.py (the FastAPI layer) so it can be invoked
directly from a script or CLI, and unit-tested without spinning up a
web server.
"""
import logging
import shutil
import tempfile
from pathlib import Path

from app.audio_utils import normalize_audio, AudioProcessingError
from app.chunking import split_audio
from app.models import TranscriptionResult
from app.transcriber import WhisperTranscriber, merge_chunk_segments

logger = logging.getLogger(__name__)


def transcribe_file(
    input_path: str,
    original_filename: str,
    transcriber: WhisperTranscriber,
    work_dir: str = None,
) -> TranscriptionResult:
    """
    Run the full pipeline on a single audio file and return a
    TranscriptionResult. Raises AudioProcessingError on invalid input.
    """
    cleanup_work_dir = work_dir is None
    work_dir = work_dir or tempfile.mkdtemp(prefix="transcribe_")
    work_dir_path = Path(work_dir)
    work_dir_path.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Starting pipeline for '%s'.", original_filename)

        # Step 1: normalize to a canonical 16kHz mono WAV, regardless of
        # the input format/extension.
        normalized_path = work_dir_path / "normalized.wav"
        info = normalize_audio(input_path, str(normalized_path))
        logger.info("Normalized '%s' -> %.2fs, %dHz, %d channel(s).",
                    original_filename, info.duration_sec, info.sample_rate, info.channels)

        # Step 2: split into chunks if the file is long; short files are
        # returned as a single "chunk" covering the whole duration.
        chunks_dir = work_dir_path / "chunks"
        chunks = split_audio(str(normalized_path), info.duration_sec, str(chunks_dir))
        logger.info("Split '%s' into %d chunk(s).", original_filename, len(chunks))

        # Step 3: transcribe all chunks. transcribe_many() lets the engine
        # decide whether to parallelize (WhisperTranscriber overlaps chunks
        # on GPU; the CPU path just goes sequentially).
        chunk_paths = [c.path for c in chunks]
        all_segments = transcriber.transcribe_many(chunk_paths)
        chunk_results = list(zip(chunks, all_segments))

        # Step 4: merge per-chunk segments into one timeline, correcting
        # timestamps and de-duplicating the overlap regions.
        merged_segments = merge_chunk_segments(chunk_results)
        logger.info("Pipeline complete for '%s': %d segments.",
                    original_filename, len(merged_segments))

        return TranscriptionResult(
            filename=original_filename,
            duration_sec=info.duration_sec,
            language=transcriber.language,
            num_chunks=len(chunks),
            segments=merged_segments,
        )
    finally:
        if cleanup_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
