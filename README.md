# Transcription Pipeline

A small FastAPI service that accepts an audio file, transcribes it to text,
and returns a JSON transcript with per-segment timestamps.

```
audio upload
  -> validate & probe (ffprobe)
  -> normalize to 16kHz mono WAV (ffmpeg)
  -> chunk if long, with overlap (ffmpeg)
  -> transcribe each chunk (faster-whisper)
  -> merge chunks into one timeline (offset + overlap dedup)
  -> JSON response
```

## Project layout

```
app/
  audio_utils.py   format detection + normalization (ffmpeg/ffprobe)
  chunking.py      splits long audio into overlapping windows
  transcriber.py   WhisperTranscriber (faster-whisper on GPU) + chunk merge
  pipeline.py      orchestrates the above into one call
  main.py          FastAPI app (the HTTP layer)
sample_output.json real output from an actual pipeline run
```

The core logic (`audio_utils`, `chunking`, `transcriber`, `pipeline`) has
**zero third-party dependencies** beyond the `ffmpeg`/`ffprobe` binaries —
FastAPI/pydantic are only used at the HTTP boundary in `main.py`.

## Running it

```bash
pip install -r requirements.txt   # ffmpeg must also be installed separately
uvicorn app.main:app --reload
```

```bash
# Transcription (downloads the 'small' Whisper model on first call)
curl -F "file=@sample_audio/short_clip.mp3" "http://localhost:8080/transcribe"
```

The model size is fixed by the `WHISPER_MODEL` env var (defaults to
`small`) and is not selectable per request.

Response shape:
```json
{
  "filename": "short_clip.mp3",
  "duration_sec": 12.0,
  "language": "en",
  "num_chunks": 1,
  "segments": [
    {"start": 0.0, "end": 4.0, "text": "Thanks for joining the call today."},
    {"start": 4.0, "end": 8.0, "text": "Let's start with a quick status update."}
  ]
}
```

## Running on GPU

Inference runs on a CUDA GPU by default. `WhisperTranscriber` is env-var
configurable, so the same code also runs on CPU by overriding the defaults:

| Env var | Default (GPU) | CPU override |
|---|---|---|
| `DEVICE` | `cuda` | `cpu` |
| `COMPUTE_TYPE` | `float16` | `int8` |
| `WHISPER_MODEL` | `small` | `small` |
| `TRANSCRIBE_WORKERS` | `2` (concurrent chunks in flight) | `1` |

**Docker (recommended)** — avoids fighting CUDA/cuDNN driver versions by hand:

```bash
docker compose up --build
# or directly:
docker build -t transcription-pipeline:gpu .
docker run --gpus all -p 8080:8080 transcription-pipeline:gpu
```

Requires the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)
on the host. Verify the GPU is visible before debugging anything else:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

**Bare metal**, if you'd rather not use Docker: install an NVIDIA driver +
CUDA 12.x + cuDNN 8.x, `pip install faster-whisper`, then run with
`uvicorn app.main:app` (GPU is already the default).

**Why `transcribe_many` matters for GPU throughput:** a single chunk
rarely saturates a GPU, so transcribing a long file's chunks one at a time
serially wastes the hardware. `WhisperTranscriber.transcribe_many()`
overlaps chunk transcription across a small thread pool (CTranslate2
releases the GIL during inference, so this is real overlap, not just Python
bookkeeping). `TRANSCRIBE_WORKERS` controls how many chunks run
concurrently — tune it against your GPU's VRAM (each concurrent chunk needs
its own working memory; 2 is a safe starting point for a 16GB card with the
`small` model).

## Logging

All stages log to stdout at `INFO` level (configured in `main.py`): request
received, upload saved, normalization, chunk counts, per-chunk
transcription, merge, and completion. Set the log level via standard Python
logging config if you need more/less detail.

---

## If I had more time

- Silence-aware chunk boundaries instead of fixed windows.
- Real text-alignment-based overlap merging instead of a timestamp
  midpoint cutoff.
- Async job queue + polling endpoint for long files instead of a
  synchronous request.
- Word-level (not just segment-level) timestamps — faster-whisper supports
  this via `word_timestamps=True`.
- Basic auth / file-size limits on the upload endpoint before exposing it
  publicly.
