#!/bin/bash

# Enable BuildKit
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# Build the image with BuildKit optimizations
docker build \
    --progress=plain \
    --no-cache \
    -t fake-news-detector:latest .

# Run the container
docker run -d \
    --name fake-news-app \
    -p 8501:8501 \
    -v $(pwd)/model:/app/model \
    fake-news-detector:latest 