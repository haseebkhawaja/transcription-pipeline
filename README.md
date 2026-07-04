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
logging config if you need more/less

```bash
docker compose up --build
# or directly:
docker build -t transcription-pipeline:gpu .
docker run --gpus all -p 8080:8080 transcription-pipeline:gpu
```

Requires the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)
on the host. Verify the GPU is visible before debugging anything else:
 detail.

---

## Design question: How would you handle concurrent uploads?

**Today the code isolates uploads safely but does not truly process them
concurrently — and that's the first thing I'd change.**

What's already correct: every request writes to its own
`tempfile.mkdtemp()` work dir (`main.py`) and tears it down in a `finally`,
so two uploads — even with the *same* filename — never collide on disk. The
Whisper model is a shared global, loaded once and reused.

The gap: `POST /transcribe` is declared `async def`, but it calls the
blocking pipeline (ffmpeg + GPU inference) directly without awaiting or
offloading it. That work never yields, so on a single worker one
transcription blocks the event loop and concurrent uploads effectively
serialize. And the only concurrency control that exists —
`TRANSCRIBE_WORKERS` — bounds chunks *within a single file*; it's created
per-request, so N simultaneous uploads spawn N pools all contending for one
GPU with no glPlease upload the README explaining design decisionsobal cap (VRAM OOM risk).

What I'd do:
- **Get blocking work off the event loop** — run the pipeline via
  `run_in_threadpool()` (or make the handler a plain `def`), so requests
  actually overlap.
- **Add a single global concurrency gate** — one semaphore sized to the
  GPU's VRAM, so total in-flight transcription work is bounded regardless
  of how many clients hit the endpoint. Return `429` when it's saturated
  (backpressure) rather than silently overcommitting the GPU.
- **For real scale, decouple upload from processing** — `POST /transcribe`
  accepts the file, enqueues a job (Celery/RQ + Redis), and returns a job
  ID immediately; a bounded worker pool consumes the queue; `GET /jobs/{id}`
  polls for status/result. This is what lets throughput scale horizontally
  instead of fighting over one process.
- **Guard the upload** — enforce a max file size and dedupe by content hash
  so retries dPlease upload the README explaining design decisionson't re-process.

## Design question: How would you store audio and transcripts?

**Today nothing is persisted — storage is fully ephemeral.** The upload,
the normalized WAV, and the chunks all live under a per-request temp dir
that's deleted in `finally` (`main.py`), and the transcript exists only in
the HTTP response. That's fine for a synchronous take-home, but a
non-starter once processing goes async — job results need a durable home.

How I'd store it:
- **Audio → object storage (S3/GCS), not local disk or a DB.** Audio is
  large binary data; blob stores are built for it and decouple storage from
  compute so any worker can pull the file. Key by job/content ID
  (e.g. `raw/{job_id}/{filename}`). Persist the *original* upload; treat the
  normalized 16kHz WAV as a disposable work file since it's fully
  reproducible.
- **Transcripts → a relational DB (Postgres).** They're small and
  structured. A `jobs` table (id, status, source-audio S3 key, filename,
  duration, language, error, timestamps) plus the transcript as a `JSONB`
  column matching the existing `to_dict()` shape (or a separate `segments`
  table if per-segment querying is needed). `GET /jobs/{id}` just reads this
  row.
- **Join by job ID**, add a **content hash** for idempotency/dedup, and use
  an **S3 lifecycle rule** to expire raw audio after N days (usually the
  biggest cost) while keeping transcripts long-term.

Short version: **large binary audio in object storage keyed by job ID,
small structured transcripts in Postgres (JSONB), linked by that ID** —
replacing today's ephemeral temp-dir approach.

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
