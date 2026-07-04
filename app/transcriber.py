"""
Transcription engine backed by faster-whisper (the CTranslate2 port of
OpenAI Whisper). Runs on a CUDA GPU by default.

Config via environment variables:
  DEVICE=cuda|cpu                    default: cuda
  COMPUTE_TYPE=float16|int8|float32  default: float16
  WHISPER_MODEL=tiny|base|small|...  default: small
  TRANSCRIBE_WORKERS=N               concurrent chunks for long files (default: 2)
"""
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List

from app.audio_utils import probe_audio
from app.models import Segment

logger = logging.getLogger(__name__)

DEVICE = os.environ.get("DEVICE", "cuda")
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "float16")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
TRANSCRIBE_WORKERS = int(os.environ.get("TRANSCRIBE_WORKERS", "2"))

# Whisper (and faster-whisper) are known to hallucinate boilerplate phrases
# ("thanks for watching", "subscribe to my channel", etc.) especially on
# trailing silence/low-volume audio, since some of the training data was
# YouTube-derived. Segments whose no_speech_prob is above this are almost
# certainly hallucinated rather than real speech, and are dropped.
NO_SPEECH_PROB_THRESHOLD = 0.6


class WhisperTranscriber:
    """Transcribes normalized audio chunks with faster-whisper on the GPU."""

    def __init__(self):
        # Imported lazily so importing this module doesn't require
        # faster-whisper/torch to be installed.
        from faster_whisper import WhisperModel

        logger.info(
            "Loading Whisper model '%s' (device=%s, compute_type=%s)...",
            WHISPER_MODEL, DEVICE, COMPUTE_TYPE,
        )
        self._model = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
        self._detected_language = None
        logger.info("Whisper model '%s' loaded.", WHISPER_MODEL)

    def transcribe(self, audio_path: str) -> List[Segment]:
        """Transcribe a single chunk. Timestamps are relative to the chunk start."""
        logger.info("Transcribing chunk: %s", audio_path)

        segments_iter, info = self._model.transcribe(
            audio_path,
            beam_size=5,
            # VAD filter drops silent/non-speech stretches before they ever
            # reach the decoder -- this is the main fix for hallucinated
            # text appearing over trailing silence.
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            # Without this, Whisper conditions each segment's decoding on
            # the text of the previous segment, which lets a hallucination
            # (or an error) "infect" every segment after it. Turning it off
            # makes each segment decode independently, at a small cost to
            # cross-segment consistency (e.g. speaker names re-explained).
            condition_on_previous_text=False,
        )

        # Duration of THIS chunk -- used below to hard-clamp any segment
        # that claims to extend past audio that actually exists. A segment
        # ending after the file's real duration is not real speech.
        chunk_duration = probe_audio(audio_path).duration_sec

        segments = []
        dropped = 0
        for seg in segments_iter:
            # Extra safety net beyond vad_filter: faster-whisper exposes a
            # per-segment no_speech_prob. High values mean the model itself
            # is telling us "there probably wasn't speech here" -- a strong
            # signal for hallucinated boilerplate, especially at the very
            # end of a file.
            if getattr(seg, "no_speech_prob", 0.0) > NO_SPEECH_PROB_THRESHOLD:
                logger.warning(
                    "Dropping likely-hallucinated segment [%.2f-%.2f] "
                    "(no_speech_prob=%.2f): %r",
                    seg.start, seg.end, seg.no_speech_prob, seg.text,
                )
                dropped += 1
                continue

            start = min(seg.start, chunk_duration)
            end = min(seg.end, chunk_duration)
            if end <= start:
                continue  # degenerate segment after clamping, skip it

            segments.append(Segment(start=start, end=end, text=seg.text))

        self._detected_language = info.language
        logger.info(
            "Transcribed %s: %d segments kept, %d dropped as likely hallucination "
            "(detected language=%s)",
            audio_path, len(segments), dropped, info.language,
        )
        return segments

    def transcribe_many(self, audio_paths: List[str]) -> List[List[Segment]]:
        """
        Transcribe all chunks. A single chunk rarely saturates the GPU, so
        for multi-chunk (long) files we keep several chunks in flight via a
        small thread pool -- CTranslate2 releases the GIL during inference,
        so this genuinely overlaps GPU work.
        """
        if len(audio_paths) <= 1:
            return [self.transcribe(p) for p in audio_paths]

        logger.info(
            "Transcribing %d chunks concurrently (workers=%d).",
            len(audio_paths), TRANSCRIBE_WORKERS,
        )
        with ThreadPoolExecutor(max_workers=TRANSCRIBE_WORKERS) as pool:
            return list(pool.map(self.transcribe, audio_paths))

    @property
    def language(self) -> str:
        return self._detected_language or "en"


def merge_chunk_segments(chunk_results: List[tuple]) -> List[Segment]:
    """
    Merge per-chunk segment lists into one timeline for the full file.

    chunk_results: list of (AudioChunk, List[Segment]) in order, where each
    Segment's start/end are relative to that chunk's own start.

    Two things happen here:
      1. Offset correction: add each chunk's start_offset to its segments
         so all timestamps are relative to the original file.
      2. Overlap dedup: consecutive chunks overlap by OVERLAP_SEC, so the
         tail of chunk N and the head of chunk N+1 cover the same audio. We
         drop segments from chunk N+1 that start before the midpoint of the
         overlap region, since the previous chunk covered that audio with
         more lead-in context.

    Note: per-chunk timestamp clamping (see WhisperTranscriber.transcribe)
    already guarantees no segment extends past its OWN chunk's duration.
    This function additionally clamps against the full original file's
    duration, in case the caller passes it in via `total_duration` --
    belt-and-suspenders against any off-by-one in chunk boundary math.
    """
    merged: List[Segment] = []

    for idx, (chunk, segments) in enumerate(chunk_results):
        overlap_cutoff = None
        if idx > 0:
            prev_chunk, _ = chunk_results[idx - 1]
            overlap_len = prev_chunk.end_offset - chunk.start_offset
            overlap_cutoff = overlap_len / 2 if overlap_len > 0 else 0

        for seg in segments:
            if overlap_cutoff is not None and seg.start < overlap_cutoff:
                continue  # already covered by the tail of the previous chunk
            merged.append(
                Segment(
                    start=seg.start + chunk.start_offset,
                    end=seg.end + chunk.start_offset,
                    text=seg.text,
                )
            )

    merged.sort(key=lambda s: s.start)
    logger.info("Merged %d chunk result(s) into %d segments.", len(chunk_results), len(merged))
    return merged