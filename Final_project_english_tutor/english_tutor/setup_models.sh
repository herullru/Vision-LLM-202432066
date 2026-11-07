#!/usr/bin/env bash
# 영어 TTS 목소리(Piper) 다운로드 - 한국어 목소리/Whisper/Gemma/YOLO는 수업에서 이미 준비됨
set -e
HOME_DIR="${VISION_LLM_HOME:-$HOME/vision-llm}"
cd "$HOME_DIR"

# 노트북에서 만든 Piper 가상환경 사용
.piper_venv/bin/python -m piper.download_voices en_US-lessac-medium --data-dir src/models/Piper

mkdir -p src/audio src/json
echo "완료: $HOME_DIR/src/models/Piper/en_US-lessac-medium.onnx"
