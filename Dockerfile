FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y python3.10 python-is-python3 pip ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app
RUN pip install -r requirements.txt
RUN pip install --no-cache-dir -U openmim
RUN mim install "mmcv>=2.0.1"
RUN mim install "mmdet>=3.1.0"
RUN mim install "mmpose>=1.1.0"

COPY . /app

EXPOSE 7860

ENV CLI_ARGS=""

CMD python run.py ${CLI_ARGS}
