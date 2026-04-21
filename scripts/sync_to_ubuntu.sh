#!/usr/bin/env bash
# macmini → Ubuntu 데이터 동기화 (이미지·라벨).
# 필요한 환경변수: DATASET_ROOT, UBUNTU_USER, UBUNTU_HOST, REMOTE_DATASET_ROOT

set -euo pipefail

: "${DATASET_ROOT:?DATASET_ROOT 환경변수 필요}"
: "${UBUNTU_USER:?UBUNTU_USER 환경변수 필요}"
: "${UBUNTU_HOST:?UBUNTU_HOST 환경변수 필요}"
: "${REMOTE_DATASET_ROOT:?REMOTE_DATASET_ROOT 환경변수 필요}"

echo "Syncing $DATASET_ROOT → $UBUNTU_USER@$UBUNTU_HOST:$REMOTE_DATASET_ROOT"

rsync -avh --progress --partial \
  --exclude '__pycache__' --exclude '.DS_Store' \
  "$DATASET_ROOT/" \
  "$UBUNTU_USER@$UBUNTU_HOST:$REMOTE_DATASET_ROOT/"

echo "Sync complete."
