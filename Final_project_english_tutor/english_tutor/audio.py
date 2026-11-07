"""음성 입출력: arecord + whisper.cpp (STT), Piper + aplay (TTS).

노트북 06번 C 파트의 subprocess 방식을 그대로 가져왔습니다.
- 한국어 문장은 한국어 Piper 목소리, 영어 단어/문장은 영어 Piper 목소리로 읽습니다.
  (한국어 목소리는 영어 발음이 좋지 않기 때문입니다.)
"""
import re
import subprocess
import wave

import numpy as np

import config as cfg

# "이게 뭐야?", "이거 뭐예요", "이건 영어로 뭐야" 등 아이다운 질문 패턴
_QUESTION_RE = re.compile(r"뭐|뭔|뭘|무엇|무슨|what", re.IGNORECASE)
# whisper가 무음에서 만들어내는 흔한 환각/태그
_NOISE_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)|♪|감사합니다|시청해 주셔서|구독")


# ---------------------------------------------------------------- STT ------
def record(seconds: int = cfg.RECORD_SECONDS) -> None:
    cfg.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "arecord", "-q",
        "-D", cfg.MIC_DEVICE,
        "-f", "S16_LE", "-r", "16000", "-c", "1",
        "-d", str(seconds),
        str(cfg.AUDIO_IN),
    ]
    if cfg.USE_PASUSPENDER:
        cmd = ["pasuspender", "--"] + cmd
    subprocess.run(cmd, check=True)


def is_silent(path=cfg.AUDIO_IN, threshold: float = cfg.SILENCE_RMS) -> bool:
    """조용한 구간은 whisper에 보내지 않는다 (속도 + 환각 방지)."""
    with wave.open(str(path), "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if data.size == 0:
        return True
    rms = float(np.sqrt(np.mean(data.astype(np.float32) ** 2)))
    return rms < threshold


def transcribe(path=cfg.AUDIO_IN) -> str:
    result = subprocess.run(
        [
            cfg.WHISPER_PATH,
            "-m", cfg.WHISPER_MODEL,
            "-f", str(path),
            "-l", "ko",
            "--no-timestamps",
            "--prompt", "이게 뭐야?",   # 아이의 질문 형태로 인식 유도
        ],
        text=True, capture_output=True, check=True,
    )
    return clean_text(result.stdout)


def clean_text(text: str) -> str:
    text = _NOISE_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_question(text: str) -> bool:
    return bool(text) and bool(_QUESTION_RE.search(text))


def listen_for_question() -> str:
    """한 번 듣고, 질문처럼 보이면 인식된 텍스트를, 아니면 빈 문자열을 반환."""
    record()
    if is_silent():
        return ""
    text = transcribe()
    return text if is_question(text) else ""


# ---------------------------------------------------------------- TTS ------
def _synth(text: str, model: str, out_path, length_scale: float = None) -> None:
    base = [cfg.PIPER_PYTHON, "-m", "piper", "-m", model, "-f", str(out_path)]
    try:
        if length_scale:
            subprocess.run(base + ["--length-scale", str(length_scale), "--", text],
                           check=True, capture_output=True)
            return
    except subprocess.CalledProcessError:
        pass  # piper 버전에 따라 옵션이 없을 수 있어 옵션 없이 재시도
    subprocess.run(base + ["--", text], check=True)


def _play(path) -> None:
    cmd = ["aplay", "-q"]
    if cfg.SPEAKER_DEVICE:
        cmd += ["-D", cfg.SPEAKER_DEVICE]
    subprocess.run(cmd + [str(path)], check=True)


def speak_ko(text: str) -> None:
    if not text:
        return
    _synth(text, cfg.PIPER_KO_MODEL, cfg.AUDIO_OUT_KO)
    _play(cfg.AUDIO_OUT_KO)


def speak_en(text: str, slow: bool = True) -> None:
    if not text:
        return
    _synth(text, cfg.PIPER_EN_MODEL, cfg.AUDIO_OUT_EN, cfg.EN_LENGTH_SCALE if slow else None)
    _play(cfg.AUDIO_OUT_EN)
