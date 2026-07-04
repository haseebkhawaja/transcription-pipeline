import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional

from app.audio_utils import AudioProcessingError
from app.pipeline import transcribe_file
from app.transcriber import WhisperTranscriber

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Transcription Pipeline", version="1.0.0")

# Load the model once and reuse it -- loading weights is expensive.
_transcriber = None


class SegmentOut(BaseModel):
    start: float
    end: float
    text: str


class TranscriptionResponse(BaseModel):
    filename: str
    duration_sec: float
    language: Optional[str]
    num_chunks: int
    segments: List[SegmentOut]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/device-info")
def device_info():
    """
    Quick way to confirm the GPU is actually visible to the process before
    debugging why transcription 'feels slow'. Doesn't load a model.
    """
    from app.transcriber import DEVICE, COMPUTE_TYPE, WHISPER_MODEL

    info = {
        "configured_device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "model": WHISPER_MODEL,
        "cuda_visible": False,
        "cuda_device_count": 0,
    }
    try:
        import ctranslate2
        info["cuda_device_count"] = ctranslate2.get_cuda_device_count()
        info["cuda_visible"] = info["cuda_device_count"] > 0
    except Exception as e:
        info["ctranslate2_error"] = str(e)
    logger.info("Device info requested: %s", info)
    return info


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    logger.info("Received transcription request for file: %s", file.filename)

    # Persist the upload to a temp file; UploadFile is a stream and the
    # pipeline (ffmpeg) needs a real path.
    work_dir = tempfile.mkdtemp(prefix="upload_")
    try:
        input_path = Path(work_dir) / file.filename
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info("Saved upload to %s", input_path)

        try:
            result = transcribe_file(
                input_path=str(input_path),
                original_filename=file.filename,
                transcriber=_get_transcriber(),
                work_dir=str(Path(work_dir) / "pipeline"),
            )
        except AudioProcessingError as e:
            logger.error("Audio processing failed for '%s': %s", file.filename, e)
            raise HTTPException(status_code=422, detail=str(e))

        logger.info(
            "Completed transcription for '%s': %.2fs, %d chunks, %d segments.",
            file.filename, result.duration_sec, result.num_chunks, len(result.segments),
        )
        return JSONResponse(content=result.to_dict())

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _get_transcriber() -> WhisperTranscriber:
    global _transcriber
    if _transcriber is None:
        try:
            _transcriber = WhisperTranscriber()
        except ImportError:
            logger.exception("faster-whisper is not installed.")
            raise HTTPException(
                status_code=500,
                detail="faster-whisper is not installed. See requirements.txt.",
            )
    return _transcriber
