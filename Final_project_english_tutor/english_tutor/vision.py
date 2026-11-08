"""YOLO 탐지 + 어떤 사물을 보고 있는지 안정적으로 결정하는 로직."""
from collections import Counter, deque
from dataclasses import dataclass
from typing import List, Optional

import cv2

import config as cfg
from objects import KID_OBJECTS


@dataclass
class Detection:
    name: str          # COCO 클래스 이름 (예: "cup")
    conf: float
    bbox: tuple        # (x1, y1, x2, y2)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


class Detector:
    def __init__(self):
        from ultralytics import YOLO

        self.model = YOLO("yolo11n.pt", task="detect")
        # 아이에게 알려줄 클래스만 탐지하도록 제한
        names = self.model.names
        self.class_ids = [i for i, n in names.items() if n in KID_OBJECTS]

    def detect(self, frame) -> List[Detection]:
        results = self.model.predict(
            source=frame,
            conf=cfg.YOLO_CONF,
            iou=cfg.YOLO_IOU,
            classes=self.class_ids,
            verbose=False,
            device="cpu,"
        )
        result = results[0]
        out = []
        for box in result.boxes:
            name = result.names[int(box.cls[0].item())]
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().tolist()]
            out.append(Detection(name, float(box.conf[0].item()), (x1, y1, x2, y2)))
        return out


def pick_main_object(dets: List[Detection], frame_shape) -> Optional[Detection]:
    """여러 개가 보일 때 아이가 보여주는 물건을 고른다.
    큰 것 + 화면 중앙에 가까운 것 + 확신도 높은 것에 점수를 준다."""
    if not dets:
        return None
    h, w = frame_shape[:2]
    cx, cy = w / 2, h / 2

    def score(d: Detection) -> float:
        x1, y1, x2, y2 = d.bbox
        dx = ((x1 + x2) / 2 - cx) / w
        dy = ((y1 + y2) / 2 - cy) / h
        center = 1.0 - min(1.0, (dx * dx + dy * dy) ** 0.5 * 2)
        return d.conf * 0.4 + (d.area / (w * h)) * 2.0 + center * 0.6

    return max(dets, key=score)


class StableTarget:
    """최근 N프레임에서 가장 자주 선택된 사물을 '지금 보고 있는 사물'로 본다.
    (손에 든 물건이 한두 프레임 깜빡여도 엉뚱한 답이 나가지 않도록)"""

    def __init__(self):
        self.history = deque(maxlen=cfg.STABLE_FRAMES)

    def update(self, det: Optional[Detection]):
        self.history.append(det.name if det else None)

    def current(self) -> Optional[str]:
        names = [n for n in self.history if n]
        if not names:
            return None
        name, count = Counter(names).most_common(1)[0]
        return name if count >= cfg.STABLE_MIN_COUNT else None


def draw(frame, dets: List[Detection], main: Optional[Detection]):
    """화면에 영어 단어를 큼직하게 표시 (cv2.putText는 한글 미지원이라 영어만 사용)."""
    for d in dets:
        x1, y1, x2, y2 = d.bbox
        is_main = main is not None and d is main
        color = (0, 200, 0) if is_main else (200, 200, 200)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if is_main else 1)
        label = KID_OBJECTS[d.name][0]
        cv2.putText(frame, f"{label} {d.conf:.2f}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame
