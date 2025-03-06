#!/usr/bin/env bash
set -v

DOCKER_IMAGE="$(grep "^ARG BASE_IMAGE=" Dockerfile | cut -f2 -d"=")"

# Run Docker command with proper formatting
docker run \
  --entrypoint /bin/bash \
  -v "$(pwd):/app" \
  -it "${DOCKER_IMAGE}" \
  -c 'cd /app && \
      pip install -U pdm && \
      pip install -U pdm && \
      export PDM_CHECK_UPDATE=false && \
      pdm install && \
      pdm export -o requirements.txt --without-hashes'
