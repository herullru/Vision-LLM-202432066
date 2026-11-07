"""유아용 사물 인식 영어 선생님 (Jetson Orin Nano)

흐름:  카메라 -> YOLO(보이는 사물 파악)
       아이 음성 "이게 뭐야?" -> Whisper(STT) -> Gemma(영어 단어 + 쉬운 설명) -> Piper(TTS)

스레드 구성 (노트북 06번에서 배운 '기능 분리' 개념을 한 프로그램 안에서 적용)
  - 메인 스레드  : 카메라 + YOLO + 화면 표시   (LLM이 돌아도 영상은 멈추지 않음)
  - 음성 스레드  : 마이크 듣기 -> STT -> 질문이면 요청 발행
  - 수업 스레드  : 요청 받으면 Gemma로 수업 생성 -> TTS로 말하기

실행:
  python main.py --dry-run --no-voice   # 소리/마이크 없이 카메라+Gemma만 확인 ('l' 키)
  python main.py                        # 전체 실행 (음성 질문 + 'l' 키)
  python main.py --no-voice             # 마이크 없이 'l' 키로만 질문
  python main.py --headless             # 모니터 없이 실행 (SSH 접속 시, Ctrl+C로 종료)
"""
import argparse
import os
import queue
import shutil
import threading
import time
from pathlib import Path

import cv2

import audio
import config as cfg
from tutor import Tutor
from vision import Detector, StableTarget, draw, pick_main_object


def find_missing(args):
    """실행에 필요한 파일/프로그램이 없으면 목록으로 알려준다. (Jetson 환경 점검)"""
    needed = {
        "YOLO 모델(.engine)": cfg.YOLO_MODEL_PATH,
        "Gemma 모델(.gguf)": cfg.GEMMA_MODEL_PATH,
    }
    if not args.dry_run:                      # 말하기에 필요
        needed["Piper 실행 Python (.piper_venv)"] = cfg.PIPER_PYTHON
        needed["Piper 한국어 목소리"] = cfg.PIPER_KO_MODEL
        needed["Piper 영어 목소리 (setup_models.sh)"] = cfg.PIPER_EN_MODEL
    if not args.no_voice:                     # 듣기에 필요
        needed["whisper-cli"] = cfg.WHISPER_PATH
        needed["Whisper 모델"] = cfg.WHISPER_MODEL

    missing = [f"{name}: {path}" for name, path in needed.items() if not Path(path).exists()]

    programs = []
    if not args.no_voice:
        programs.append("arecord")
        if cfg.USE_PASUSPENDER:
            programs.append("pasuspender")
    if not args.dry_run:
        programs.append("aplay")
    missing += [f"프로그램 없음: {p}" for p in programs if shutil.which(p) is None]
    return missing


class App:
    def __init__(self, args):
        self.args = args
        self.stop = threading.Event()
        self.speaking = threading.Event()          # TTS 재생 중에는 마이크를 끈다 (에코 방지)
        self.requests = queue.Queue(maxsize=1)     # 질문이 쌓이지 않도록 1개만
        self.target = StableTarget()
        self.target_lock = threading.Lock()
        self.dry_run = args.dry_run

        print("[1/4] 필요한 파일 확인...")
        missing = find_missing(args)
        if missing:
            print("아래 항목을 찾을 수 없습니다. 경로(config.py)나 설치를 확인하세요:")
            for item in missing:
                print("  -", item)
            raise SystemExit(1)

        print("[2/4] YOLO 로딩...")
        self.detector = Detector()
        print("[3/4] Gemma 로딩... (처음엔 시간이 걸려요)")
        self.tutor = Tutor()
        print("[4/4] 준비 완료!")

    # ------------------------------------------------------------ 출력 ----
    def say_ko(self, text):
        print(f"[선생님-KO] {text}")
        if not self.dry_run:
            self.speaking.set()
            try:
                audio.speak_ko(text)
            finally:
                self.speaking.clear()

    def say_en(self, text, slow=True):
        print(f"[선생님-EN] {text}")
        if not self.dry_run:
            self.speaking.set()
            try:
                audio.speak_en(text, slow)
            finally:
                self.speaking.clear()

    # ------------------------------------------------------- 수업 스레드 --
    def teach_loop(self):
        while not self.stop.is_set():
            try:
                self.requests.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.teach_once()
            except Exception as exc:
                print(f"[오류] 수업 중 문제 발생: {exc}")
            time.sleep(0.5)

    def teach_once(self):
        with self.target_lock:
            coco_name = self.target.current()

        if coco_name is None:
            self.say_ko("무슨 물건인지 잘 안 보여요. 카메라에 가까이 보여 주세요!")
            return

        lesson = self.tutor.lesson_for(coco_name)
        if lesson is None:
            self.say_ko("음, 이건 아직 잘 모르겠어요. 다른 물건을 보여 주세요!")
            return

        self.say_ko(f"{lesson.ko_name}! 영어로는 {lesson.word}!")
        self.say_en(lesson.en_sentence)
        self.say_ko("다 함께 크게 따라 해 보세요!")
        time.sleep(1.0)

    # ------------------------------------------------------- 음성 스레드 --
    def listen_loop(self):
        while not self.stop.is_set():
            if self.speaking.is_set() or not self.requests.empty():
                time.sleep(0.2)
                continue
            try:
                text = audio.listen_for_question()
            except Exception as exc:
                print(f"[오류] 음성 인식 문제: {exc}")
                time.sleep(1.0)
                continue
            if text:
                print(f"[아이] {text}")
                self.request()

    def request(self):
        try:
            self.requests.put_nowait(True)
        except queue.Full:
            pass  # 이미 수업 요청이 대기 중

    # ------------------------------------------------------ 메인 스레드 --
    def open_camera(self):
        """Jetson의 USB 웹캠은 V4L2로 먼저 열고, 안 되면 기본 방식으로 다시 시도."""
        cap = cv2.VideoCapture(cfg.CAMERA_INDEX, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(cfg.CAMERA_INDEX)
        return cap

    def run(self):
        cap = self.open_camera()
        if not cap.isOpened():
            print("카메라를 열 수 없습니다. (ls /dev/video* 로 장치를 확인하고 CAMERA_INDEX를 바꿔 보세요)")
            return

        threading.Thread(target=self.teach_loop, daemon=True).start()
        if not self.args.no_voice:
            threading.Thread(target=self.listen_loop, daemon=True).start()

        has_display = bool(os.environ.get("DISPLAY"))
        show = (not self.args.headless) and has_display
        if not show and not self.args.headless:
            print("DISPLAY가 없어 영상 창 없이 실행합니다. (SSH 접속이면 정상)")
        print("영상 창에서  l : '이게 뭐야?' 요청 | q : 종료  (창이 없으면 Ctrl+C)")
        if not self.dry_run:
            threading.Thread(
                target=lambda: self.say_ko("안녕! 나는 영어 선생님이야. 물건을 보여 주고 이게 뭐야? 하고 물어봐!"),
                daemon=True,
            ).start()

        try:
            while not self.stop.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                dets = self.detector.detect(frame)
                main = pick_main_object(dets, frame.shape)
                with self.target_lock:
                    self.target.update(main)

                if show:
                    cv2.imshow("English Teacher", draw(frame, dets, main))
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("l"):
                        self.request()
                else:
                    time.sleep(0.01)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop.set()
            cap.release()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="유아용 사물 인식 영어 선생님 (Jetson)")
    parser.add_argument("--no-voice", action="store_true", help="마이크 끄기 ('l' 키로만 질문)")
    parser.add_argument("--headless", action="store_true", help="화면 없이 실행")
    parser.add_argument("--dry-run", action="store_true", help="소리 없이 텍스트만 출력 (디버깅)")
    App(parser.parse_args()).run()


if __name__ == "__main__":
    main()
