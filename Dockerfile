FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Avoid interactive tzdata prompt during apt install.
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY app/ app/

# GPU defaults. Override at `docker run -e ...` time as needed.
ENV DEVICE=cuda
ENV COMPUTE_TYPE=float16
ENV WHISPER_MODEL=small
ENV TRANSCRIBE_WORKERS=2

EXPOSE 8080

# Pre-download the model into the image at build time so the first
# request doesn't pay a cold-start download. Comment this out if you'd
# rather keep the image small and download at first request instead.
# RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('${WHISPER_MODEL}', device='cpu')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
