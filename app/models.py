"""
Core data models for the transcription pipeline.

These are plain dataclasses (not pydantic) so that the core pipeline logic
(audio_utils, chunking, transcriber, merging) has zero third-party
dependencies and can be unit-tested without installing FastAPI/pydantic.
The FastAPI layer (main.py) wraps these in pydantic models at the boundary.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Segment:
    """A single transcribed segment with timestamps, in seconds."""
    start: float
    end: float
    text: str

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "text": self.text.strip(),
        }


@dataclass
class AudioInfo:
    """Metadata about a probed audio file."""
    path: str
    duration_sec: float
    sample_rate: int
    channels: int
    codec: str
    format_name: str


@dataclass
class TranscriptionResult:
    """Final output of the pipeline for one uploaded file."""
    filename: str
    duration_sec: float
    language: Optional[str]
    num_chunks: int
    segments: List[Segment] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "duration_sec": round(self.duration_sec, 2),
            "language": self.language,
            "num_chunks": self.num_chunks,
            "segments": [s.to_dict() for s in self.segments],
        }
