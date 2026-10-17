import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.data

# 1. 이미지 불러오기
image_path = 'src/images/basketball_crop.jpg'
basketball = cv2.imread(image_path)

# 시각화를 위해 원본 BGR 이미지를 RGB로 변환하여 복사
final_img = cv2.cvtColor(basketball.copy(), cv2.COLOR_BGR2RGB)

# 2. LAB 색공간 변환 및 각 채널 분리
basketball_lab = cv2.cvtColor(basketball, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(basketball_lab)

# 3. CLAHE 생성 (clipLimit 3.0, tileGridSize 8x8) 및 L 채널 강화
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
l_clahe_enhanced = clahe.apply(l)
basketball_enhanced_lab = cv2.merge((l_clahe_enhanced, a, b))

# 4. LAB 범위 설정 및 강화된 이미지로 binary mask(mask_enhanced) 생성
lower_basketball_lab = np.array([70, 130, 70])
upper_basketball_lab = np.array([180, 230, 185])
mask_enhanced = cv2.inRange(basketball_enhanced_lab, lower_basketball_lab, upper_basketball_lab)

# 5. Morphological operation(Opening)을 적용하여 최종 binary mask 생성
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
final_mask = cv2.morphologyEx(mask_enhanced, cv2.MORPH_OPEN, kernel)

# 6. Binary mask로 외곽선(Contour) 검출
contours, _ = cv2.findContours(
    final_mask,              # binary mask
    cv2.RETR_EXTERNAL,       # 바깥쪽 테두리만 검출
    cv2.CHAIN_APPROX_SIMPLE  # 꼭짓점 좌표만 저장
)

# 7. 최종 농구공 영역 선택 및 시각화 오버레이
if len(contours) > 0:
    # 가장 면적이 큰 contour 선택
    largest_contour = max(contours, key=cv2.contourArea)

    # 가장 큰 contour만 채운 binary mask 생성
    filled_mask = np.zeros_like(final_mask)
    cv2.drawContours(
        filled_mask,
        [largest_contour],
        -1,
        255,
        thickness=cv2.FILLED
    )
    
    # 반투명 초록색 마스크 오버레이 적용
    mask_color = np.zeros_like(final_img)
    mask_color[:, :, 1] = filled_mask
    alpha = 0.35
    final_img = cv2.addWeighted(final_img, 1.0, mask_color, alpha, 0)

    # 초록색 외곽선 그리기
    cv2.drawContours(final_img, [largest_contour], -1, (0, 255, 0), 2)

# 8. 결과 출력
plt.imshow(final_img)
plt.axis("off")
plt.show()