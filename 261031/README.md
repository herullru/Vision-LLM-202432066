# 딥러닝 기초 및 GPU 가속 / 객체 탐지 (IoU, NMS) 과제

`03_DL-and-GPU.ipynb`와 `04_DL-Object-Detection.ipynb`의 실습 중 아래 내용을 구현했습니다.

* `03_DL-and-GPU.ipynb`
  * GPU 가속 (이미지 잔상 효과, Convolution2D)
  * MNIST 데이터셋 `main_gpu()` 함수
  * CIFAR-10 데이터셋 모델 학습 파트 + 저장한 모델 파일 (`.pth`)
* `04_DL-Object-Detection.ipynb`
  * IoU, NMS 구현

## 파일 구조 (필수)

* `dl_gpu_assignment.py`: MNIST CNN 학습 결과 시각화(Loss/Accuracy 그래프, Feature Map)와 CuPy GPU 가속 실습(이미지 잔상 효과 `motion_trail_np`/`motion_trail_cp`, Convolution2D `convolution2d_cp`)의 NumPy(CPU) vs. CuPy(GPU) 처리 시간 비교
* `mnist_gpu.py`: MNIST 데이터셋 CPU 학습(`main_cpu()`)과 GPU 학습(`main_gpu()`)의 학습 시간 비교
* `cifar10_gpu.py`: CIFAR-10 데이터셋 CNN 모델(`CIFAR10CNN`)의 GPU 학습 및 평가, 학습 지표 시각화, 모델 저장/불러오기
* `iou_nms.py`: Bounding Box IoU 계산(`calculate_iou`), Confidence Thresholding(`prune_low_conf`), NMS(`apply_nms`) 구현 및 OpenCV `cv2.dnn.NMSBoxes`와의 결과 비교
* `src/models/CIFAR10/CIFAR10_CNN.pth`: `cifar10_gpu.py` 실행 후 생성되는 학습된 CIFAR-10 모델 파일

## 실행 방식 (필수)

### 환경

* Python 3
* 공통: `pip install numpy matplotlib opencv-python`
* PyTorch 계열: `pip install torch torchvision`
* GPU 가속(CuPy) 실습: NVIDIA GPU와 CUDA 버전에 맞는 `cupy`
* IoU/NMS: `pip install scikit-image`
* 이미지와 데이터셋은 실행하는 폴더 기준 `src/` 아래에 두거나 내려받습니다.
  * `src/images/seagull.jpg`: 잔상 효과, Convolution 실습용 이미지
  * `src/images/two_apples1.png` ~ `two_apples3.png`: IoU 실습용 이미지
  * `src/datasets/`: MNIST, CIFAR-10 (없으면 `download=True`로 자동 다운로드)

### 실행

```bash
# 1) GPU 가속 (잔상 효과, Convolution2D) + MNIST CNN 시각화
python dl_gpu_assignment.py

# 2) MNIST CPU vs. GPU 학습 시간 비교 (main_gpu)
python mnist_gpu.py

# 3) CIFAR-10 GPU 학습 + 모델 저장(.pth)
python cifar10_gpu.py

# 4) IoU, NMS
python iou_nms.py
```

### 참고

* `dl_gpu_assignment.py`의 잔상 효과(CPU) 처리는 Python 반복문으로 픽셀을 하나씩 계산하므로 시간이 오래 걸릴 수 있습니다.
* `dl_gpu_assignment.py`는 CuPy 또는 NVIDIA GPU가 없으면 GPU 실습 부분을 건너뜁니다. `mnist_gpu.py`와 `cifar10_gpu.py`는 CUDA GPU가 없으면 `RuntimeError`가 발생합니다.
* 그림 창이 뜨면 창을 닫아야 다음 단계가 진행됩니다.
* `cifar10_gpu.py`는 학습이 끝나면 `src/models/CIFAR10/CIFAR10_CNN.pth`에 모델 가중치, optimizer 상태, 마지막 테스트 정확도, epoch 수, 클래스 이름, 정규화 값(mean/std)을 저장하고 다시 불러와 예측 결과를 확인합니다.

## 구현한 내용

### GPU 가속

* **이미지 잔상 효과:** 이미지를 오른쪽으로 `shift`칸 이동한 복사본에 가중치를 곱해 누적합니다. CPU(`motion_trail_np`)는 출력 픽셀을 반복문으로 하나씩 계산하고, GPU(`motion_trail_cp`)는 이동 횟수만 반복하며 배열 슬라이싱으로 전체 픽셀 연산을 병렬 처리합니다.
* **Convolution2D:** `convolution2d_cp`는 커널 원소를 하나씩 선택해 모든 출력 위치의 대응 픽셀 배열에 곱해 누적합니다. 기존 `convolution2d`(CPU)와 결과를 비교하고 처리 시간을 측정합니다.
* **시간 측정:** GPU 연산은 비동기로 실행되므로 `cp.cuda.Stream.null.synchronize()`(CuPy), `torch.cuda.synchronize()`(PyTorch)로 동기화한 뒤 시간을 측정합니다.

### MNIST `main_gpu()`

* `main_cpu()`와 동일한 데이터셋, 모델, 학습/평가 흐름에서 장치만 `cuda`로 지정하고 시간 측정 전후에 `torch.cuda.synchronize()`를 호출합니다.
* 학습 시간을 반환해 CPU 학습 시간과 비교(시간 차이, 속도 향상 배율)합니다.

### CIFAR-10 모델 학습

* 데이터 증강(RandomCrop, RandomHorizontalFlip)과 채널별 정규화를 적용합니다.
* `Conv → BatchNorm → ReLU` 두 번과 MaxPool로 구성된 `ConvBlock` 3개(64/128/256 채널), Global Average Pooling, Dropout, Linear로 `CIFAR10CNN`을 구성했습니다.
* CUDA GPU에서 5 epoch 동안 학습과 평가를 수행하고 epoch별/전체 학습 시간을 출력합니다.
* 학습 후 Loss/Accuracy 그래프와 예측 결과를 확인하고, 모델을 `.pth`로 저장한 뒤 다시 불러옵니다.

### IoU, NMS

* **IoU:** `(x, y, w, h)` BBox를 `(x1, y1, x2, y2)`로 변환(`convert_bbox_coord`)한 뒤 교집합/합집합 면적으로 IoU를 계산하며, 합집합이 0이면 0을 반환해 `ZeroDivisionError`를 방지합니다.
* **NMS:** Confidence Thresholding 후, Confidence가 가장 높은 BBox를 선택하고 IoU가 임계값을 넘는 BBox를 제거하는 과정을 반복합니다. 결과를 OpenCV `cv2.dnn.NMSBoxes` 결과와 비교합니다.

수업 따라가는 게 너무 벅차지만 새로운 걸 배워서 신기해요
이걸로 만족할래요...