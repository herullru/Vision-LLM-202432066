"""04_DL-Object-Detection: B. IoU / C. NMS 실습
실행: python iou_nms.py
필요 패키지: numpy, matplotlib, opencv-python, scikit-image
IoU 실습 이미지: 실행하는 폴더 기준 src/images/two_apples1~3.png
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import skimage.data
from random import randint, gauss


# ===================== BBox 그리기 함수 =====================
def draw_bbox(img, coord1, coord2, color, thickness=3, txt=""):
    _, txt_h = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.rectangle(img, coord1, coord2, color, thickness)
    cv2.putText(img, txt, (coord1[0], coord1[1]-txt_h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


# ===================== B. IoU =====================
def convert_bbox_coord(bbox):
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def calculate_iou(bbox_a, bbox_b):
    xa_1, ya_1, xa_2, ya_2 = bbox_a
    xb_1, yb_1, xb_2, yb_2 = bbox_b

    intersection_x1 = max(xa_1, xb_1)
    intersection_y1 = max(ya_1, yb_1)
    intersection_x2 = min(xa_2, xb_2)
    intersection_y2 = min(ya_2, yb_2)

    w = max(0, intersection_x2 - intersection_x1)
    h = max(0, intersection_y2 - intersection_y1)
    intersection_area = w * h
    union_area = (xa_2 - xa_1) * (ya_2 - ya_1) + (xb_2 - xb_1) * (yb_2 - yb_1) - intersection_area

    # TODO: ZeroDivisionError 주의
    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def get_contour(img_bgr):
    red_lower1 = np.array([0, 100, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 100, 50])
    red_upper2 = np.array([180, 255, 255])
    green_lower = np.array([25, 10, 30])
    green_upper = np.array([100, 255, 255])

    img_blr = cv2.GaussianBlur(img_bgr, (7, 7), 0)
    blr_hsv = cv2.cvtColor(img_blr, cv2.COLOR_BGR2HSV)

    red_mask1 = cv2.inRange(blr_hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(blr_hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    grn_mask = cv2.inRange(blr_hsv, green_lower, green_upper)

    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_grn, _ = cv2.findContours(grn_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cntr_red, cntr_grn = None, None

    if len(contours_red) > 0:
        cntr_red = max(contours_red, key=cv2.contourArea)
    if len(contours_grn) > 0:
        cntr_grn = max(contours_grn, key=cv2.contourArea)

    return cntr_red, cntr_grn


def run_iou():
    apples1 = cv2.imread("src/images/two_apples1.png")
    apples2 = cv2.imread("src/images/two_apples2.png")
    apples3 = cv2.imread("src/images/two_apples3.png")

    apples1_rgb = cv2.cvtColor(apples1, cv2.COLOR_BGR2RGB)
    apples2_rgb = cv2.cvtColor(apples2, cv2.COLOR_BGR2RGB)
    apples3_rgb = cv2.cvtColor(apples3, cv2.COLOR_BGR2RGB)

    bgr = [apples1, apples2, apples3]
    rgb = [apples1_rgb, apples2_rgb, apples3_rgb]

    ious = []
    for i in range(3):
        cntr_red, cntr_grn = get_contour(bgr[i])
        bbox1 = cv2.boundingRect(cntr_red)
        bbox2 = cv2.boundingRect(cntr_grn)

        x_r1, y_r1, x_r2, y_r2 = convert_bbox_coord(bbox1)
        x_g1, y_g1, x_g2, y_g2 = convert_bbox_coord(bbox2)

        draw_bbox(rgb[i], (x_r1, y_r1), (x_r2, y_r2), (255, 0, 0), txt="Red Apple")
        draw_bbox(rgb[i], (x_g1, y_g1), (x_g2, y_g2), (0, 255, 0), txt="Green Apple")

        iou = calculate_iou(convert_bbox_coord(bbox1), convert_bbox_coord(bbox2))
        print(f"IoU {i + 1}: {iou}")
        ious.append(iou)

    plt.figure(figsize=(16, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1), plt.imshow(rgb[i]), plt.title(f"IoU: {ious[i]:.4f}")
    for ax in plt.gcf().axes:
        ax.axis("off")
    plt.show()


# ===================== C. NMS =====================
from random import randint, gauss

def random_bbox(img, min_w=100, min_h=400, max_w=600, max_h=600):
    img_h, img_w = img.shape[:2]

    w = randint(min_w, min(max_w, img_w))
    h = randint(min_h, min(max_h, img_h))

    x1 = randint(0, img_w - w)
    y1 = randint(0, img_h - h)

    return x1, y1, x1 + w, y1 + h

def random_conf():
    return min(1.0, max(0.0, gauss(0.6, 0.1)))


def prune_low_conf(boxes, scores, conf_threshold=0.5):
    """
    Confidence Thresholding 적용

    Parameters
    ----------
    boxes : list
        Bounding Box 목록
        각 박스는 (x1, y1, x2, y2) 형식

    scores : list
        각 Bounding Box의 Confidence Score

    conf_threshold : float
        Bounding Box를 유지하기 위한 최소 Confidence Score

    Returns
    -------
    pruned_boxes : list
        Confidence Score가 임계값 이상인 Bounding Box 목록

    pruned_scores : list
        Confidence Score가 임계값 이상인 Bounding Box의 Confidence Score 목록
    """
    pruned_boxes = []
    pruned_scores = []

    for box, score in zip(boxes, scores):
        if score >= conf_threshold:
            pruned_boxes.append(box)
            pruned_scores.append(score)

    return pruned_boxes, pruned_scores


def apply_nms(boxes, scores, iou_threshold=0.5):
    """
    NMS 적용

    Parameters
    ----------
    boxes : list
        Bounding Box 목록
        각 박스는 (x1, y1, x2, y2) 형식

    scores : list
        각 Bounding Box의 Confidence Score

    iou_threshold : float
        중복 Bounding Box를 제거할 IoU 기준값

    Returns
    -------
    nms_boxes : list
        NMS 적용 후 유지된 Bounding Box 목록

    nms_scores : list
        NMS 적용 후 유지된 Bounding Box의 Confidence Score 목록
    """
    if len(boxes) == 0:
        return [], []

    # Confidence가 높은 순서로 인덱스 정렬
    order = np.argsort(np.array(scores))[::-1]

    keep = []

    while order.size > 0:
        # 남은 박스 중 Confidence가 가장 높은 박스 선택
        best = order[0]
        keep.append(best)

        # 나머지 박스와 IoU 계산
        rest = order[1:]
        ious = np.array([calculate_iou(boxes[best], boxes[j]) for j in rest])

        # IoU가 임계값 이하인 박스만 남김 (초과하면 제거)
        order = rest[ious <= iou_threshold]

    nms_boxes = [boxes[i] for i in keep]
    nms_scores = [scores[i] for i in keep]

    return nms_boxes, nms_scores


def apply_nms_cv2(boxes, scores, conf_threshold=0.5, iou_threshold=0.5):
    """
    OpenCV의 cv2.dnn.NMSBoxes()를 이용해 NMS 적용

    Parameters
    ----------
    boxes : list
        Bounding Box 목록
        각 박스는 (x1, y1, x2, y2) 형식

    scores : list
        각 Bounding Box의 Confidence Score

    conf_threshold : float
        Bounding Box를 유지하기 위한 최소 Confidence Score

    iou_threshold : float
        중복 Bounding Box를 제거할 IoU 기준값

    Returns
    -------
    nms_boxes : list
        NMS 적용 후 유지된 Bounding Box 목록
        각 박스는 기존과 동일한 (x1, y1, x2, y2) 형식

    nms_scores : list
        NMS 적용 후 유지된 Bounding Box의 Confidence Score
    """

    if len(boxes) == 0:
        return [], []

    # cv2.dnn.NMSBoxes()는 (x, y, width, height) 형식을 사용
    boxes_xywh = []

    for box in boxes:
        x1, y1, x2, y2 = box

        width = x2 - x1
        height = y2 - y1

        boxes_xywh.append([int(x1), int(y1), int(width), int(height)])

    # OpenCV NMS 적용하여 BBox 제거 후 남은 BBox의 인덱스
    keep_indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xywh,
        scores=scores,
        score_threshold=conf_threshold,
        nms_threshold=iou_threshold,
    )

    # 1차원 리스트로 변환
    keep_indices = np.array(keep_indices).reshape(-1).tolist()

    # 원래 xyxy 형식의 박스에서 선택
    nms_boxes = [boxes[index] for index in keep_indices]
    nms_scores = [scores[index] for index in keep_indices]

    return nms_boxes, nms_scores


def run_nms():
    # ----- 랜덤 BBox 생성
    hubble = skimage.data.hubble_deep_field()

    N = 20

    boxes = []
    scores = []

    for i in range(N):
        boxes.append(random_bbox(hubble))
        scores.append(random_conf())

    for i, box in enumerate(boxes):
        draw_bbox(
            hubble,
            (box[0], box[1]),
            (box[2], box[3]),
            (randint(0, 255), randint(0, 255), randint(0, 255)),
            txt=f"{scores[i]:.2f}"
        )

    plt.figure(figsize=(10, 8))
    plt.imshow(hubble)
    plt.axis("off")
    plt.title("Before NMS")
    plt.show()

    # ----- 직접 구현한 Confidence Thresholding + NMS
    print(len(boxes), len(scores))

    new_boxes, new_scores = prune_low_conf(boxes, scores)
    new_boxes, new_scores = apply_nms(new_boxes, new_scores, iou_threshold=0.2)

    print(len(new_boxes), len(new_scores))

    hubble_custom_nms = skimage.data.hubble_deep_field()

    for i, box in enumerate(new_boxes):
        draw_bbox(
            hubble_custom_nms,
            (box[0], box[1]),
            (box[2], box[3]),
            (randint(0, 255), randint(0, 255), randint(0, 255)),
            txt=f"{new_scores[i]:.2f}"
        )

    plt.figure(figsize=(10, 8))
    plt.imshow(hubble_custom_nms)
    plt.axis("off")
    plt.title("Custom NMS")
    plt.show()

    # ----- OpenCV NMS
    print(len(boxes), len(scores))

    cv2_new_boxes, cv2_new_scores = apply_nms_cv2(boxes, scores, iou_threshold=0.2)

    print(len(cv2_new_boxes), len(cv2_new_scores))

    hubble_cv2_nms = skimage.data.hubble_deep_field()

    for i, box in enumerate(cv2_new_boxes):
        draw_bbox(
            hubble_cv2_nms,
            (box[0], box[1]),
            (box[2], box[3]),
            (randint(0, 255), randint(0, 255), randint(0, 255)),
            txt=f"{cv2_new_scores[i]:.2f}"
        )

    # ----- 결과 비교
    plt.figure(figsize=(16, 10))

    plt.subplot(1, 2, 1), plt.imshow(hubble_custom_nms), plt.title("Custom NMS Result")
    plt.subplot(1, 2, 2), plt.imshow(hubble_cv2_nms), plt.title("CV2 NMS Result")

    for ax in plt.gcf().axes:
        ax.axis("off")
    plt.show()

    print("Custom == CV2 (박스 목록 동일 여부):", sorted(new_boxes) == sorted(cv2_new_boxes))


if __name__ == "__main__":
    run_iou()
    run_nms()