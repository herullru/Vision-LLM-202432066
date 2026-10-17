# Basketball Object Detection & Morphological Processing Assignment

# Basketball Object Detection and Morphological Processing Assignment

본 과제는 OpenCV와 scikit-image 라이브러리를 활용하여 이미지 내 특정 객체인 농구공을 LAB 색공간 변환, CLAHE 명암비 강화, 이진 마스크 생성, 모폴로지 연산 및 컨투어 검출을 통해 탐지하고 시각화하는 파이썬 프로그램입니다.

## 1. 필수 의존성

프로그램 실행을 위해 아래의 파이썬 라이브러리가 설치되어 있어야 합니다.

```bash
pip install opencv-python numpy matplotlib scikit-image

```

## 2. 폴더 구조

프로젝트 디렉토리는 다음과 같은 구조로 배치되어야 합니다.

```text
261017/
├── assignment2.py       # 실행할 메인 파이썬 코드
└── src/
    └── images/
        └── basketball_crop.jpg  # 실습 대상 이미지 파일

```

## 3. 실행 방법

1. 터미널(Command Prompt 또는 VS Code 통합 터미널)을 엽니다.
2. 프로젝트가 위치한 폴더로 이동합니다.
3. 아래 명령어를 입력하여 코드를 실행합니다.

```bash
python assignment2.py

```

## 4. 주요 파이프라인 설명

1. **이미지 로드 및 색공간 변환**: OpenCV를 통해 BGR 이미지를 읽어온 뒤 LAB 색공간으로 변환하고 채널을 분리합니다.
2. **CLAHE 적용**: 조명 변화에 강건하도록 L 채널에 CLAHE(Contrast Limited Adaptive Histogram Equalization)를 적용하여 화질을 개선합니다.
3. **이진 마스크 생성**: `cv2.inRange`를 사용하여 지정한 LAB 임계값 범위에 해당하는 농구공 영역을 이진 마스크로 추출합니다.
4. **모폴로지 연산 (Opening)**: 노이즈를 제거하기 위해 `cv2.MORPH_OPEN` 연산을 적용하여 마스크를 정제합니다.
5. **컨투어 및 오버레이 시각화**: `cv2.findContours`로 외곽선을 추출하고, 가장 큰 객체를 선택한 뒤 반투명 초록색 마스크와 외곽선을 원본 이미지 위에 오버레이하여 시각화합니다.

## 5. 모폴로지 연산(Opening) 선택 이유

* **클로징(Closing) 대신 오프닝(Opening)을 적용한 이유**: 클로징은 객체 내부의 끊어진 선을 메우는 데 유리하나 배경의 불필요한 바깥쪽 잡음까지 함께 팽창시킬 위험이 있습니다. 반면 오프닝 연산은 침식 후 팽창을 수행하므로, 배경의 미세한 돌출 노이즈를 효과적으로 제거하면서도 농구공 객체의 원래 크기와 형태적 특징을 손실 없이 보존할 수 있습니다.

## 6. 기타 참고 사항

* 수업 중 `src` 폴더 파일을 개인 깃허브 레포지토리에 아직 푸시하지 못하여, 현재는 과제 제출 파일 패키지 내부에 `src` 폴더를 직접 포함하여 제출합니다.
* 이후 개인 깃허브에 정상적으로 `src` 폴더를 포함한 전체 프로젝트를 푸시할 예정이며, 푸시 완료 후 코드 내 파일 경로 등을 알맞게 수정하여 동기화할 계획입니다.