import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# 영상 뒤집기: None(안 뒤집음) / 0(상하) / 1(좌우, 거울처럼) / -1(상하+좌우)
# 거울처럼 보이는 게 손 움직임과 직관적으로 맞아서 기본값은 1
FLIP_CODE = 1

# 초록색 범위 (LAB 색공간, [L, a, b] 순서)
GREEN_LOWER = np.array([30, 60, 90], dtype=np.uint8)
GREEN_UPPER = np.array([230, 115, 180], dtype=np.uint8)

# 이 면적(픽셀)보다 작은 덩어리는 노이즈로 보고 무시
MIN_AREA = 500

# 기준점과 물체 중심 사이 거리가 (물체 반지름 x 이 값)보다 작으면 "가깝다"고 판단
# 고정 픽셀 대신 반지름 비율을 써서, 손/물체가 카메라에 가깝든 멀든 비슷하게 동작
GRAB_RATIO = 1.5

# "가깝다"가 연속으로 이 프레임 수만큼 이어져야 Grabbed (깜빡임 방지)
GRAB_FRAMES = 5
# Grabbed 상태에서 "멀다"가 연속으로 이 프레임 수만큼 이어져야 해제
RELEASE_FRAMES = 5

# 마스크(흑백) 확인용 창을 같이 띄울지
SHOW_MASK = True


# =====================================================================
# 손 인식 모델 준비
# =====================================================================

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")


def ensure_model():
    """모델 파일이 없으면 인터넷에서 내려받는다."""
    if not os.path.exists(MODEL_PATH):
        print("[모델] hand_landmarker.task 내려받는 중... (처음 한 번만)")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[모델] 다운로드 완료")


# 손 관절 번호 연결 정보 (손 뼈대를 그릴 때 사용)
# 0=손목, 1~4=엄지, 5~8=검지, 9~12=중지, 13~16=약지, 17~20=소지
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
THUMB_TIP = 4
INDEX_TIP = 8


# =====================================================================
# 카메라 열기 (젯슨 YUYV 전용)
# =====================================================================

# 젯슨용 GStreamer 파이프라인 (수업 노트북 내용)
# /dev/video0이 아니라면 device= 뒤의 번호를 바꾸세요. (ls /dev/video* 로 확인)
PIPELINE_YUYV = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

def open_camera():
    """YUYV 카메라를 열고, 실제로 프레임이 읽히면 cap을 돌려준다."""
    cap = cv2.VideoCapture(PIPELINE_YUYV, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        ret, _ = cap.read()  # 진짜로 한 장 읽히는지 확인
        if ret:
            return cap
    cap.release()
    return None


# =====================================================================
# 초록 물체 찾기
# =====================================================================

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def find_green_object(frame):
    """
    가장 큰 초록 덩어리를 찾는다.
    찾으면 (중심 좌표, 반지름, 마스크), 못 찾으면 (None, None, 마스크)
    """
    blurred = cv2.GaussianBlur(frame, (7, 7), 0)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    mask = cv2.inRange(lab, GREEN_LOWER, GREEN_UPPER)

    # Opening(작은 노이즈 제거) -> Closing(구멍 메우기)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > MIN_AREA]
    if len(contours) == 0:
        return None, None, mask

    largest = max(contours, key=cv2.contourArea)
    (_, _), radius = cv2.minEnclosingCircle(largest)
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None, None, mask
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), int(radius), mask


# =====================================================================
# 손 찾기
# =====================================================================

def find_hand(detector, frame):
    """
    손을 찾아서 (21개 관절의 픽셀 좌표 리스트, 엄지-검지 중점)을 돌려준다.
    손이 없으면 (None, None)
    """
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # MediaPipe는 RGB를 원함
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    if len(result.hand_landmarks) == 0:
        return None, None

    hand = result.hand_landmarks[0]
    # landmark 좌표는 0~1 비율이라 가로/세로 길이를 곱해 픽셀 좌표로 바꾼다
    points = [(int(p.x * w), int(p.y * h)) for p in hand]

    thumb = np.array(points[THUMB_TIP])
    index = np.array(points[INDEX_TIP])
    pinch_point = tuple(((thumb + index) // 2).astype(int))
    return points, pinch_point


def draw_hand(frame, points):
    """손 뼈대(선)와 관절(점)을 그린다. 엄지/검지 끝은 빨간 점으로 강조."""
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (0, 255, 0), 2)
    for i, p in enumerate(points):
        if i in (THUMB_TIP, INDEX_TIP):
            cv2.circle(frame, p, 7, (0, 0, 255), -1)
        else:
            cv2.circle(frame, p, 4, (255, 0, 0), -1)


# =====================================================================
# 메인 루프
# =====================================================================

def main():
    ensure_model()

    # 모델 파일을 파이썬이 직접 읽어서 넘긴다.
    with open(MODEL_PATH, "rb") as f:
        model_data = f.read()

    # 손 인식기 만들기 (손 1개만)
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_buffer=model_data),
        num_hands=1,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = open_camera()
    if cap is None:
        print("카메라를 열지 못했어요. 카메라 연결과 /dev/video 번호를 확인하세요.")
        return

    grabbed = False   # 현재 Grabbed 상태인지
    near_count = 0    # "가깝다"가 연속으로 몇 프레임 이어졌는지
    far_count = 0     # "멀다"가 연속으로 몇 프레임 이어졌는지

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽지 못해 종료합니다.")
            break

        # 뒤집기는 여기서 한 번만! 이후 물체/손 모두 같은 프레임을 사용
        if FLIP_CODE is not None:
            frame = cv2.flip(frame, FLIP_CODE)

        # 물체와 손 찾기
        obj_center, obj_radius, mask = find_green_object(frame)
        hand_points, pinch_point = find_hand(detector, frame)

        # 손 그리기
        if hand_points is not None:
            draw_hand(frame, hand_points)
            cv2.circle(frame, pinch_point, 8, (255, 255, 255), 2)  # 기준점(흰 원)

        # 물체 그리기
        if obj_center is not None:
            cv2.circle(frame, obj_center, obj_radius, (255, 0, 0), 2)
            cv2.circle(frame, obj_center, 5, (255, 0, 0), -1)

        # 손과 물체가 둘 다 있을 때만 거리 계산
        status = ""
        if obj_center is not None and hand_points is not None:
            # 두 점 사이 유클리드 거리
            distance = np.linalg.norm(np.array(obj_center) - np.array(pinch_point))
            is_near = distance < obj_radius * GRAB_RATIO

            # 연속 프레임 카운트로 Grabbed 상태 결정 (깜빡임 방지)
            if is_near:
                near_count += 1
                far_count = 0
            else:
                far_count += 1
                near_count = 0

            if not grabbed and near_count >= GRAB_FRAMES:
                grabbed = True
            elif grabbed and far_count >= RELEASE_FRAMES:
                grabbed = False

            # 선 + 거리 텍스트
            line_color = (0, 255, 0) if grabbed else (0, 255, 255)
            cv2.line(frame, pinch_point, obj_center, line_color, 2)
            mid = ((pinch_point[0] + obj_center[0]) // 2,
                   (pinch_point[1] + obj_center[1]) // 2)
            cv2.putText(frame, f"{distance:.0f}px", mid,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, line_color, 2)

            status = "Grabbed!" if grabbed else "Tracking"
        else:
            # 손이나 물체가 안 보이면 카운터 초기화
            near_count = 0
            far_count += 1
            if grabbed and far_count >= RELEASE_FRAMES:
                grabbed = False

            if obj_center is None and hand_points is None:
                status = "No hand, no object"
            elif obj_center is None:
                status = "No green object"
            else:
                status = "No hand"

        # 상태 표시 (왼쪽 위)
        color = (0, 255, 0) if grabbed else (0, 255, 255)
        size = 1.4 if grabbed else 0.9
        cv2.putText(frame, status, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, size, color, 3 if grabbed else 2)

        cv2.imshow("Hand & Object Grab Detector", frame)
        if SHOW_MASK:
            cv2.imshow("Green Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()