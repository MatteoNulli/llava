ARG BASE_IMAGE=hub.tess.io/krylov_curated_workspace/base:cuda12.2-py3.10-ubuntu20.04@sha256:2c76c84c195ba4259659415efc76eaaa8d692cce6e1542c4f0f90c8915b2cb3a

FROM --platform="${TARGETPLATFORM:-linux/amd64}" ${BASE_IMAGE}
RUN sudo apt update -y && sudo apt upgrade -y

ARG PYPROJECT=pyproject.toml
RUN pip install --no-cache-dir --upgrade pip

WORKDIR /lmms-eval
COPY requirements.txt /lmms-eval/requirements.txt
RUN pip install -r requirements.txt
RUN pip install pychomsky==0.2.20 --extra-index-url https://artifactory.corp.ebay.com/artifactory/api/pypi/pypi-coreai/simple
ENV ENV_CHOMSKY_KRYLOV_WORKSPACE=1
