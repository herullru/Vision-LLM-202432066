"""설정 파일 - 경로/장치 번호는 환경 변수로 덮어쓸 수 있습니다.

기본 경로는 수업 노트북(05, 06번)에서 사용한 ~/vision-llm 구조를 그대로 따릅니다.
"""
import os
from pathlib import Path

# 수업에서 사용한 프로젝트 루트 (src/, whisper.cpp/, .piper_venv/ 가 있는 곳)
HOME = Path(os.getenv("VISION_LLM_HOME", Path.home() / "vision-llm"))

# ---- 모델 경로 -------------------------------------------------------------
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", str(HOME / "src/models/YOLO/yolo11n_int8.engine"))
GEMMA_MODEL_PATH = os.getenv("GEMMA_MODEL_PATH", str(HOME / "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"))

WHISPER_PATH = os.getenv("WHISPER_PATH", str(HOME / "whisper.cpp/build-cpu/bin/whisper-cli"))
WHISPER_MODEL = os.getenv("WHISPER_MODEL", str(HOME / "whisper.cpp/models/ggml-base.bin"))

PIPER_PYTHON = os.getenv("PIPER_PYTHON", str(HOME / ".piper_venv/bin/python"))
PIPER_KO_MODEL = os.getenv("PIPER_KO_MODEL", str(HOME / "src/models/Piper/ko_KR-kss-medium.onnx"))
PIPER_EN_MODEL = os.getenv("PIPER_EN_MODEL", str(HOME / "src/models/Piper/en_US-lessac-medium.onnx"))

# ---- 오디오 ----------------------------------------------------------------
# `arecord -l` 로 확인한 마이크 번호 (노트북 예시: plughw:3,0)
MIC_DEVICE = os.getenv("MIC_DEVICE", "plughw:2,0")
# 스피커를 따로 지정하려면 예: plughw:0,3  (비워두면 시스템 기본 장치)
SPEAKER_DEVICE = os.getenv("SPEAKER_DEVICE", "")
USE_PASUSPENDER = os.getenv("USE_PASUSPENDER", "1") == "1"   # 노트북의 pasuspender 사용 여부

RECORD_SECONDS = int(os.getenv("RECORD_SECONDS", "4"))       # 한 번에 듣는 시간
SILENCE_RMS = float(os.getenv("SILENCE_RMS", "300"))         # 이 값보다 조용하면 STT 생략
EN_LENGTH_SCALE = float(os.getenv("EN_LENGTH_SCALE", "1.2")) # 1.0보다 크면 천천히 읽음 (아이용)

AUDIO_DIR = HOME / "src/audio"
AUDIO_IN = AUDIO_DIR / "input.wav"
AUDIO_OUT_KO = AUDIO_DIR / "tutor_ko.wav"
AUDIO_OUT_EN = AUDIO_DIR / "tutor_en.wav"

# ---- 카메라 / YOLO ----------------------------------------------------------
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.35"))
YOLO_IOU = 0.5
STABLE_FRAMES = 12      # 최근 N프레임 중
STABLE_MIN_COUNT = 5    # 같은 사물이 M번 이상 보이면 '확실히 보임'

# ---- Gemma ------------------------------------------------------------------
CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "1024"))  # 짧은 프롬프트라 작게 -> 메모리 절약
MAX_TOKENS = 80
LESSON_CACHE = HOME / "src/json/lesson_cache.json"          # 한 번 만든 설명은 저장해서 재사용
