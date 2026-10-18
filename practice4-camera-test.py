import cv2
pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)


cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)


cap.release()
cv2.destroyAllWindows()

import numpy as np
green_lower = np.array([30, 60, 90], dtype=np.uint8)
green_upper = np.array([230, 115, 180], dtype=np.uint8)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

while True:
    ret, frame = cap.read()
    
    if not ret:
            break
    if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    output_frame = frame.copy()

    frame_lab=cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(frame_lab)
    l_clahe = clahe.apply(l)
    enhanced_lab = cv2.merge((l_clahe, a, b))
    mask = cv2.inRange(enhanced_lab, green_lower, green_upper)
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
    opened_mask,        # binary mask
    cv2.RETR_EXTERNAL,       # 구멍 뚫린 객체의 바깥쪽 테두리만 검출
    cv2.CHAIN_APPROX_SIMPLE  # 직선 경로 위에 있는 픽셀 좌표 전체 대신 꼭짓점 좌표만 저장
)
    final_img_enhanced = frame.copy()

    if len(contours) > 0:
    # 가장 큰 contour
        largest_contour = max(contours, key=cv2.contourArea)

    # 가장 큰 contour만 채운 binary mask 생성
    largest_mask = np.zeros_like(opened_mask)
    cv2.drawContours(
        largest_mask,
        [largest_contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    # 가장 큰 contour만 채운 binary mask를 반투명 빨간색으로 overlay
    mask_color = np.zeros_like(final_img_enhanced)
    mask_color[:, :, 2] = largest_mask
    alpha = 0.15
    final_img_enhanced = cv2.addWeighted(final_img_enhanced, 1.0, mask_color, alpha, 0)

    # Contour 그리기 (빨간색 외곽선)
    cv2.drawContours(final_img_enhanced, [largest_contour], -1, (255,0,0), 2)

    # 최소 외접원 중심 및 반지름
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    center_circle = (int(x), int(y))
    radius = int(radius)

    cv2.imshow("VideoCapture", final_img_enhanced)

cap.release()
cv2.destroyAllWindows()