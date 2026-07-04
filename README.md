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
sample_audio/      sample files (short mp3/wav)
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

## Design question: How do you handle different audio formats?

**Normalize everything to one canonical format immediately, before
anything else touches the file.** Specifically: 16kHz, mono, 16-bit PCM
WAV, via `ffmpeg`. This is the format Whisper-family models expect
internally anyway, so doing it up front means every downstream stage
(chunking, transcription) is format-agnostic and only has to deal with one
shape of input.

Concretely:
- **Detection isn't extension-based.** File extensions lie (mislabeled
  files, no extension at all). `ffprobe` sniffs the actual container/codec
  from file bytes, and that's what actually decides whether a file is
  usable — the extension check in the API layer is just a fast, friendly
  pre-check, not the source of truth.
- **Validation happens before conversion.** `probe_audio()` confirms the
  file has a readable audio stream and a determinable duration. If it
  doesn't (corrupt file, video-only file, truncated upload), we reject with
  a clear 422 error instead of letting it fail deep inside the STT model
  with a confusing stack trace.
- **Channel/sample-rate normalization is one ffmpeg call:** `-ac 1`
  downmixes any channel count to mono, `-ar 16000` resamples to 16kHz,
  `-sample_fmt s16` standardizes bit depth. Stereo, 44.1/48kHz files, mp3,
  m4a, ogg, flac — all converge to the same canonical WAV.

## Design question: How do you deal with long audio files?

Three separate concerns, handled independently:

**1. Splitting the file.** Files under a threshold (15 min) are treated as
a single chunk — no need to add complexity for short files. Longer files
are split into fixed 10-minute windows with a 2-second overlap between
consecutive windows, so words spoken right at a cut point aren't lost.
I chose fixed-length windows over silence-based splitting for
predictability and simplicity; a silence-aware splitter (via ffmpeg's
`silencedetect` filter) is a natural upgrade if boundary cuts prove to be a
real accuracy problem in practice.

**2. Merging chunks back into one timeline.** Each chunk's segments are
offset by that chunk's start time in the original file. The overlap region
between consecutive chunks would otherwise produce duplicate text — I drop
segments starting before the midpoint of the overlap window, since the
previous chunk had more lead-in context for that audio. This is a
deliberate simplification: it's not real text alignment, so a sentence
that straddles the cutoff exactly could occasionally get truncated or
duplicated. A more robust version would fuzzy-match text in the overlap
window and splice at the best match — noted here rather than over-building
it for a take-home.

**3. Not blocking the request for the full transcription time.** This
implementation is synchronous for simplicity, but for real long files (a
3-hour podcast) I would *not* hold an HTTP connection open the whole time.
The change I'd make: `POST /transcribe` kicks off a background job and
immediately returns a job ID; `GET /jobs/{id}` polls for status/result
(e.g. via a task queue like Celery/RQ, or FastAPI `BackgroundTasks` for a
lighter-weight version). Independent chunks are also naturally
parallelizable — separate workers or batched GPU inference — since chunk
transcription has no cross-chunk dependency until the merge step.

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
