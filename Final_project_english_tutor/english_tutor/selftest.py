"""하드웨어(Jetson/마이크/카메라) 없이 돌릴 수 있는 로직 테스트.  python selftest.py"""
import tempfile
from pathlib import Path

import audio
import config as cfg
from tutor import Tutor, fallback_lesson, parse_lesson


class FakeLLM:
    def __init__(self, text):
        self.text = text
        self.calls = 0

    def create_chat_completion(self, **kwargs):
        self.calls += 1
        return {"choices": [{"message": {"content": self.text}}]}


def test_question_detection():
    assert audio.is_question("이게 뭐야?")
    assert audio.is_question("이거 영어로 뭐예요")
    assert audio.is_question("What is this")
    assert not audio.is_question("")
    assert not audio.is_question("안녕하세요 오늘 날씨 좋다")
    assert audio.clean_text("[BLANK_AUDIO] 이게  뭐야 ") == "이게 뭐야"


def test_parse_good_output():
    raw = "EN: I drink from a cup.\nKO: 컵에는 물이나 우유를 담아 마셔요."
    lesson = parse_lesson(raw, "cup", "컵")
    assert lesson.en_sentence == "I drink from a cup."
    assert lesson.ko_text.startswith("컵에는")


def test_parse_messy_output():
    raw = "**EN:** I read a book! 📖\n**KO:** 책을 읽으면 재미있는 이야기를 알 수 있어요 ✨"
    lesson = parse_lesson(raw, "book", "책")
    assert lesson.en_sentence == "I read a book!"
    assert "📖" not in lesson.en_sentence and "✨" not in lesson.ko_text


def test_parse_bad_output_falls_back():
    # 단어 누락 / 한국어가 영어 줄에 섞임 / 형식 불량
    for raw in ["그냥 설명해 드릴게요", "EN: I like it.\nKO: 좋아요", "EN: 컵이에요 cup\nKO: 컵"]:
        lesson = parse_lesson(raw, "cup", "컵")
        assert "cup" in lesson.en_sentence.lower()
        assert not any("가" <= ch <= "힣" for ch in lesson.en_sentence)


def test_article():
    assert fallback_lesson("apple", "사과").en_sentence == "This is an apple."
    assert fallback_lesson("cup", "컵").en_sentence == "This is a cup."


def test_cache_and_unknown():
    with tempfile.TemporaryDirectory() as d:
        cfg.LESSON_CACHE = Path(d) / "cache.json"
        llm = FakeLLM("EN: I kick the ball.\nKO: 공은 던지고 굴리며 놀아요.")
        tutor = Tutor(llm=llm)
        a = tutor.lesson_for("sports ball")
        b = tutor.lesson_for("sports ball")
        assert a == b and llm.calls == 1          # 두 번째는 캐시
        assert Tutor(llm=llm).lesson_for("sports ball") == a and llm.calls == 1  # 파일 캐시
        assert tutor.lesson_for("knife") is None   # 위험/미등록 사물


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK ", name)
    print("모든 테스트 통과")
