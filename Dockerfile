FROM python:3.9-slim

WORKDIR /app

RUN apt update && \
    apt install -y --no-install-recommends \
        libxcb1 \
        ffmpeg \
        libsm6 \
        libxext6 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install spade==3.3.3 aiofiles==23.2.1 opencv-python

RUN mkdir -p /app/received_photos

COPY . .

ENTRYPOINT ["python", "-m", "agent"]