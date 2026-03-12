#!/usr/bin/env bash
# Build and push the EI MAR RAG pipeline image.
# Set IMAGE to your registry/repo:tag before running, e.g.:
#   export IMAGE=ghcr.io/your-org/rag-ei-mar:latest
#   export IMAGE=docker.io/youruser/rag-ei-mar:latest

set -e

: "${IMAGE:?Set IMAGE to your registry/repo:tag, e.g. ghcr.io/your-org/rag-ei-mar:latest}"

echo "Building $IMAGE ..."
docker build -t "$IMAGE" .

echo "Pushing $IMAGE ..."
docker push "$IMAGE"

echo "Done. Run with: docker run --rm -v $(pwd)/Documents:/app/Documents -v $(pwd)/data:/app/data $IMAGE"
