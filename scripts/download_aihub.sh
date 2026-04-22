#!/usr/bin/env bash
# AIhub 배·사과 화상병 데이터셋 수동 다운로드 가이드 + 무결성 검증.
# 실제 다운로드는 AIhub 웹에서 로그인 후 수행해야 함.

set -euo pipefail

: "${DATASET_ROOT:?DATASET_ROOT 환경변수 필요}"
root="$DATASET_ROOT/raw/aihub"

echo "기대 경로: $root"
echo "필수 하위 구조:"
echo "  $root/pear/images/*.jpg"
echo "  $root/pear/annotations/*.{json,xml}"
echo "  $root/apple/images/*.jpg"
echo "  $root/apple/annotations/*.{json,xml}"
echo

ok=1
for crop in pear apple; do
  for kind in images annotations; do
    d="$root/$crop/$kind"
    if [[ ! -d "$d" ]]; then
      echo "❌ missing: $d"
      ok=0
      continue
    fi
    count=$(find "$d" -type f | wc -l | tr -d ' ')
    echo "✅ $d — $count 개 파일"
  done
done

if [[ "$ok" == 0 ]]; then
  echo
  echo "일부 경로 누락. AIhub에서 해당 데이터셋을 다운로드해 위 구조로 배치하세요."
  exit 1
fi

echo
echo "AIhub 디렉토리 구조 OK. 다음 단계:"
echo "  1) scripts/run_labeling.py 로 VLM 재라벨링 (macmini)"
echo "  2) scripts/sync_to_ubuntu.sh 로 Ubuntu 전송"
echo "  3) scripts/preprocess.py 로 split 생성 (Ubuntu)"
