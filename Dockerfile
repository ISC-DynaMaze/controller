FROM python:3.9-slim

WORKDIR /app

RUN apt update && \
    apt install -y --no-install-recommends \
        libxcb1 \
        ffmpeg \
        libsm6 \
        libxext6 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN mkdir -p /app/photos

COPY ./agent /app/agent

ENTRYPOINT ["python", "-m", "agent"]