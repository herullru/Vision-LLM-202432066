"""Gemma(llama.cpp)로 아이용 영어 수업 내용을 만든다.

설계 포인트
1) 영어 단어/한국어 이름은 objects.py에서 확정 -> Gemma는 '예문 + 쉬운 설명'만 생성
2) 출력 형식을 두 줄(EN:, KO:)로 고정하고, 어기면 안전한 기본 문장으로 대체
3) 한 번 만든 수업은 JSON으로 캐시 -> 같은 물건은 LLM 없이 즉시 답변
4) 이모지/마크다운 제거 (TTS가 그대로 읽어버리므로)
"""
import json
import re
from dataclasses import dataclass
from typing import Optional

import config as cfg
from objects import KID_OBJECTS


@dataclass
class Lesson:
    word: str        # 영어 단어
    ko_name: str     # 한국어 이름
    en_sentence: str # 영어 예문
    ko_text: str     # 한국어 쉬운 설명


SYSTEM_PROMPT = """\
Instruction:
너는 3~6세 한국 아이에게 영어 단어를 알려주는 다정한 영어 선생님이야.
Context에 주어진 사물에 대해 쉬운 영어 문장 하나와 쉬운 한국어 설명 한 문장을 만들어.

Constraint:
- 영어 문장은 5단어 안팎의 아주 쉬운 문장이고, 반드시 주어진 영어 단어를 그대로 포함해.
- 한국어 설명은 아이에게 말하듯 "~해요" 체로 짧게(30자 이내), 그 물건을 어디에 쓰는지 알려줘.
- 이모지, 별표, 따옴표, 괄호, 설명 외의 말은 쓰지 마.
- 아이에게 무섭거나 위험한 내용은 쓰지 마.

Output Format (정확히 두 줄):
EN: <영어 문장>
KO: <한국어 설명>

Example:
Context: 영어 단어 = ball, 한국어 = 공
EN: I kick the ball.
KO: 공은 데굴데굴 굴리고 던지며 놀아요.
"""


def _article(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" else "a"


def fallback_lesson(word: str, ko_name: str) -> Lesson:
    return Lesson(
        word=word,
        ko_name=ko_name,
        en_sentence=f"This is {_article(word)} {word}.",
        ko_text=f"{ko_name}은 영어로 {word}라고 해요.",
    )


def _strip_junk(text: str) -> str:
    # 이모지/마크다운/특수문자 제거 (한글, 영문, 숫자, 기본 문장부호만 남김)
    text = re.sub(r"[^0-9A-Za-z가-힣 .,!?']", "", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_lesson(raw: str, word: str, ko_name: str) -> Lesson:
    """모델 출력에서 EN/KO 두 줄을 뽑고, 검증에 실패하면 기본 문장으로 대체."""
    base = fallback_lesson(word, ko_name)
    en = ko = None
    for line in raw.splitlines():
        m = re.match(r"\s*\**\s*(EN|KO)\s*\**\s*[:：]\s*(.+)", line.strip(), re.IGNORECASE)
        if not m:
            continue
        key, value = m.group(1).upper(), _strip_junk(m.group(2))
        if key == "EN" and en is None:
            en = value
        elif key == "KO" and ko is None:
            ko = value

    # 영어 문장 검증: 단어 포함, 영문자 위주, 너무 길지 않게
    if not en or word.lower() not in en.lower() or re.search(r"[가-힣]", en) or len(en.split()) > 10:
        en = base.en_sentence
    # 한국어 문장 검증: 한글 포함, 너무 길지 않게
    if not ko or not re.search(r"[가-힣]", ko) or len(ko) > 60:
        ko = base.ko_text
    return Lesson(word, ko_name, en, ko)


class Tutor:
    def __init__(self, llm=None):
        """llm을 직접 넣으면(테스트용) 모델 로드를 건너뛴다."""
        if llm is None:
            from llama_cpp import Llama

            llm = Llama(
                model_path=cfg.GEMMA_MODEL_PATH,
                n_gpu_layers=-1,
                n_ctx=cfg.CONTEXT_WINDOW,
                n_batch=32,
                n_ubatch=32,
                verbose=False,
            )
        self.llm = llm
        self.cache = self._load_cache()

    # ---- 캐시 ----
    def _load_cache(self) -> dict:
        try:
            return json.loads(cfg.LESSON_CACHE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save_cache(self) -> None:
        try:
            cfg.LESSON_CACHE.parent.mkdir(parents=True, exist_ok=True)
            tmp = cfg.LESSON_CACHE.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(cfg.LESSON_CACHE)
        except OSError:
            pass

    # ---- 수업 생성 ----
    def lesson_for(self, coco_name: str) -> Optional[Lesson]:
        if coco_name not in KID_OBJECTS:
            return None
        word, ko_name = KID_OBJECTS[coco_name]

        if coco_name in self.cache:
            return Lesson(**self.cache[coco_name])

        try:
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context: 영어 단어 = {word}, 한국어 = {ko_name}"},
                ],
                max_tokens=cfg.MAX_TOKENS,
                temperature=0.3,
            )
            raw = response["choices"][0]["message"]["content"]
        except Exception as exc:  # LLM 오류가 나도 수업은 계속
            print(f"[Tutor] LLM 오류, 기본 문장 사용: {exc}")
            return fallback_lesson(word, ko_name)

        lesson = parse_lesson(raw, word, ko_name)
        self.cache[coco_name] = lesson.__dict__
        self._save_cache()
        return lesson
