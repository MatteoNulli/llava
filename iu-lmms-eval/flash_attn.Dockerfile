# This is hub.tess.io/krylov_curated_workspace/base:cuda12.1-devel-py3.10-ubuntu20.04 but using sha-tagged base image to track changes
ARG BASE_IMAGE=hub.tess.io/krylov_curated_workspace/base@sha256:a4e7120018e314c147ac4636fe8a3fc5d60acf525d13b3c15a718533a2f87ddc

FROM --platform="${TARGETPLATFORM:-linux/amd64}" ${BASE_IMAGE}
ARG PYPROJECT=pyproject.toml
RUN pip install --no-cache-dir --upgrade pip

WORKDIR /lmms-eval
COPY requirements.txt /lmms-eval/requirements.txt
RUN pip install -r requirements.txt
RUN pip install flash-attn --no-build-isolation # cannot be installed from requirements.txt
