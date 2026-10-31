"""03_DL-and-GPU 실습 (MNIST CNN 학습 결과 시각화 + CuPy GPU 가속 실습)

실행:  python dl_gpu_assignment.py

필요 패키지: numpy, matplotlib, opencv-python, torch, torchvision, cupy(GPU 실습용, 선택)
- 이미지: 이 파일과 같은 폴더 기준 src/images/seagull.jpg
- MNIST : 이 파일과 같은 폴더 기준 src/datasets (없으면 자동 다운로드)
- CuPy(NVIDIA GPU)가 없으면 GPU 실습(G 섹션)은 건너뜁니다.
- 그림 창은 닫아야 다음 단계로 넘어갑니다.
"""

import os
import time
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    import cupy as cp
    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    cp = None
    GPU_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ 전역 설정
IMAGE_PATH = os.path.join(BASE_DIR, "src", "images", "seagull.jpg")
DATASET_ROOT = os.path.join(BASE_DIR, "src", "datasets")

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
TRAIL_LENGTH = 32

EPOCHS = 3
SEED = 42

# 필터 / 편향 (Convolution 실습)
dx_edge_filter = np.array(
    [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1],
    ],
    dtype=np.float32,
)

dy_edge_filter = np.array(
    [
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1],
    ],
    dtype=np.float32,
)

dx_bias = -0.2
dy_bias = -0.2


# ------------------------------------------------------------------ 기본 함수
def relu(x):
    return np.maximum(0, x)


def convolution2d(image, kernel, bias=0.0, stride=1, padding=0):
    """2차원 이미지에 하나의 CNN 필터를 적용 (NumPy, 출력 픽셀을 하나씩 계산)."""
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    padded_image = np.pad(
        image,
        ((padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0,
    )

    output_height = (padded_image.shape[0] - kernel_height) // stride + 1
    output_width = (padded_image.shape[1] - kernel_width) // stride + 1

    output = np.zeros((output_height, output_width), dtype=np.float32)

    for output_y in range(output_height):
        for output_x in range(output_width):
            start_y = output_y * stride
            start_x = output_x * stride

            image_region = padded_image[
                start_y:start_y + kernel_height,
                start_x:start_x + kernel_width,
            ]

            weighted_sum = np.sum(image_region * kernel)
            output[output_y, output_x] = weighted_sum + bias

    return output


# ------------------------------------------------------------------ CNN 모델
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x, return_features=False):
        conv1_output = self.conv1(x)
        relu1_output = F.relu(conv1_output)
        pool1_output = self.pool(relu1_output)
        conv2_output = self.conv2(pool1_output)
        relu2_output = F.relu(conv2_output)
        pool2_output = self.pool(relu2_output)
        flattened = torch.flatten(pool2_output, start_dim=1)
        fc1_output = F.relu(self.fc1(flattened))
        logits = self.fc2(fc1_output)

        if return_features:
            features = {
                "conv1": conv1_output,
                "relu1": relu1_output,
                "pool1": pool1_output,
                "conv2": conv2_output,
                "relu2": relu2_output,
                "pool2": pool2_output,
            }
            return logits, features

        return logits


def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        predictions = logits.argmax(dim=1)
        correct_count += (predictions == labels).sum().item()
        sample_count += batch_size

    average_loss = total_loss / sample_count
    accuracy = correct_count / sample_count

    return average_loss, accuracy


def evaluate(model, data_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            predictions = logits.argmax(dim=1)
            correct_count += (predictions == labels).sum().item()
            sample_count += batch_size

    average_loss = total_loss / sample_count
    accuracy = correct_count / sample_count

    return average_loss, accuracy


def show_feature_maps(feature_tensor, title, max_maps=8, cmap="gray"):
    if feature_tensor.ndim != 4:
        raise ValueError("Feature Tensor는 [Batch, Channel, Height, Width] 형태여야 합니다.")

    feature_maps = feature_tensor[0].detach().cpu()

    channel_count = min(feature_maps.shape[0], max_maps)
    column_count = 4
    row_count = int(np.ceil(channel_count / column_count))

    plt.figure(figsize=(12, 3 * row_count))

    for channel_index in range(channel_count):
        plt.subplot(row_count, column_count, channel_index + 1)
        plt.imshow(feature_maps[channel_index], cmap=cmap)
        plt.title(f"Channel {channel_index}")
        plt.axis("off")
        plt.suptitle(title)


# ------------------------------------------------------------------ F. MNIST CNN 실습
def run_mnist_cnn():
    device = torch.device("cpu")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root=DATASET_ROOT, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=DATASET_ROOT, train=False, transform=transform, download=True)

    print("Train data:", len(train_dataset))
    print("Test data:", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=0)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ----- 학습
    epochs = EPOCHS
    train_loss_history = []
    train_accuracy_history = []
    test_loss_history = []
    test_accuracy_history = []

    training_start_time = time.perf_counter()

    for epoch in range(epochs):
        epoch_start_time = time.perf_counter()

        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)
        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)

        epoch_time = time.perf_counter() - epoch_start_time

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_accuracy * 100:.2f}% | "
            f"Test loss: {test_loss:.4f} | "
            f"Test accuracy: {test_accuracy * 100:.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )

    total_training_time = time.perf_counter() - training_start_time
    print(f"Total CPU training time: {total_training_time:.2f}s")

    # ----- 이미지 하나 추론
    image, true_label = test_dataset[0]
    input_batch = image.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(input_batch)
        probabilities = torch.softmax(logits, dim=1)
        predicted_label = probabilities.argmax(dim=1).item()

    print("True label:", true_label)
    print("Predicted label:", predicted_label)
    print("Confidence:", probabilities[0, predicted_label].item())

    plt.figure(figsize=(4, 4))
    plt.imshow(image.squeeze(0), cmap="gray")
    plt.title(f"True: {true_label}, Prediction: {predicted_label}")
    plt.axis("off")
    plt.show()

    # ----- 16) 모델 성능 확인: Loss 그래프
    epoch_range = range(1, epochs + 1)

    plt.figure(figsize=(8, 5))
    # TODO: train_loss_history 그래프 생성
    plt.plot(epoch_range, train_loss_history, marker="o", label="Train Loss")
    # TODO: test_loss_history 그래프 생성
    plt.plot(epoch_range, test_loss_history, marker="o", label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Loss")
    plt.grid()
    plt.legend()
    plt.show()

    # ----- Accuracy 그래프
    plt.figure(figsize=(8, 5))
    # TODO: train_accuracy_history 그래프 생성 (백분율)
    plt.plot(epoch_range, np.array(train_accuracy_history) * 100, marker="o", label="Train Accuracy")
    # TODO: test_accuracy_history 그래프 생성 (백분율)
    plt.plot(epoch_range, np.array(test_accuracy_history) * 100, marker="o", label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("CNN Accuracy")
    plt.grid()
    plt.legend()
    plt.show()

    # ----- 17) 학습된 모델의 Feature Map 확인
    # TODO: 이미지 선택
    image, true_label = test_dataset[0]
    input_batch = image.unsqueeze(0).to(device)

    # TODO: 역전파(파라미터 수정) 없이 순전파 수행
    model.eval()
    with torch.no_grad():
        logits, features = model(input_batch, return_features=True)

    # TODO: Conv1 Feature Maps 8개 시각화
    show_feature_maps(features["conv1"], title="Conv1 Feature Maps", max_maps=8, cmap="viridis")
    plt.show()

    # TODO: Conv2 Feature Maps 16개 시각화
    show_feature_maps(features["conv2"], title="Conv2 Feature Maps", max_maps=16, cmap="viridis")
    plt.show()

    # TODO: 6개 Layer별 대표 Feature Map 4개씩 비교
    layer_names = [
        "conv1",
        "relu1",
        "pool1",
        "conv2",
        "relu2",
        "pool2",
    ]

    plt.figure(figsize=(12, 18))
    for row, name in enumerate(layer_names):
        feature_maps = features[name][0].detach().cpu()
        for col in range(4):
            plt.subplot(len(layer_names), 4, row * 4 + col + 1)
            plt.imshow(feature_maps[col], cmap="viridis")
            plt.title(f"{name} - Channel {col}")
            plt.axis("off")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------ G-4) 이미지 잔상 효과
def motion_trail_np(image, weights):
    """NumPy: 출력 픽셀을 Python 반복문으로 하나씩 계산한다."""
    # TODO: 필요한 변수 정의 및 출력 배열 생성
    height, width = image.shape
    output = np.zeros_like(image)

    for output_y in range(height):
        for output_x in range(width):

            for shift in range(len(weights)):
                # TODO: x값 shift
                input_x = output_x - shift

                # TODO: shift한 픽셀을 누적
                # TODO: 출력 배열에 누적 픽셀 저장
                if input_x >= 0:
                    output[output_y, output_x] += weights[shift] * image[output_y, input_x]

    return output


def motion_trail_cp(image, weights):
    """CuPy: 이동 거리는 순차 반복, 각 이동 거리의 전체 배열 연산은 GPU 병렬 처리."""
    # TODO: 필요한 변수 정의 및 출력 배열 생성
    height, width = image.shape
    output = cp.zeros_like(image)

    for shift in range(len(weights)):
        # TODO: 배열 슬라이싱
        input_region = image[:, :width - shift]
        output_region = output[:, shift:]

        # TODO: GPU가 병렬 연산할 output_region 계산 코드
        output_region += weights[shift] * input_region

    return output


def sync():
    cp.cuda.Stream.null.synchronize()


def load_input_image():
    img_gray = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

    if img_gray is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {IMAGE_PATH}")

    img_gray = cv2.resize(img_gray, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
    return img_gray.astype(np.float32) / 255.0


def run_motion_trail(input_np):
    weights_np = np.linspace(1.0, 0.05, TRAIL_LENGTH, dtype=np.float32)
    weights_np /= np.sum(weights_np)

    # ----- NumPy CPU 처리 (오래 걸립니다)
    print("[잔상] CPU 처리 중... (시간이 오래 걸릴 수 있습니다)")
    cpu_start = time.perf_counter()
    output_np = motion_trail_np(image=input_np, weights=weights_np)
    cpu_end = time.perf_counter()
    cpu_time = cpu_end - cpu_start

    # ----- CuPy 배열 준비
    upload_start = time.perf_counter()
    input_cp = cp.asarray(input_np)
    weights_cp = cp.asarray(weights_np)
    sync()
    upload_end = time.perf_counter()
    upload_time = upload_end - upload_start

    # ----- GPU 워밍업
    warmup_image = cp.zeros((64, 64), dtype=cp.float32)
    warmup_weights = cp.asarray(weights_np[:4])
    _ = motion_trail_cp(image=warmup_image, weights=warmup_weights)
    sync()

    # ----- CuPy GPU 처리
    sync()
    gpu_start = time.perf_counter()
    output_cp = motion_trail_cp(image=input_cp, weights=weights_cp)
    sync()
    gpu_end = time.perf_counter()
    gpu_time = gpu_end - gpu_start

    # ----- GPU 결과를 CPU로 가져오기
    download_start = time.perf_counter()
    output_cp_np = cp.asnumpy(output_cp)
    download_end = time.perf_counter()
    download_time = download_end - download_start

    compute_speedup = cpu_time / gpu_time
    gpu_total_time = upload_time + gpu_time + download_time
    total_speedup = cpu_time / gpu_total_time

    print("========== NumPy CPU ==========")
    print(f"Processing time: {cpu_time:.4f} seconds")
    print()
    print("========== CuPy GPU ==========")
    print(f"CPU to GPU: {upload_time:.4f} seconds")
    print(f"GPU processing: {gpu_time:.4f} seconds")
    print(f"GPU to CPU: {download_time:.4f} seconds")
    print(f"Total GPU time: {gpu_total_time:.4f} seconds")
    print()
    print("========== Comparison ==========")
    print(f"Compute-only speedup: {compute_speedup:.2f}x")
    print(f"Including transfers: {total_speedup:.2f}x")
    print(f"Results match: {np.allclose(output_np, output_cp_np, atol=1e-5)}")

    plt.figure(figsize=(18, 12))
    plt.subplot(1, 3, 1), plt.imshow(input_np, cmap="gray"), plt.title("Original Image")
    plt.subplot(1, 3, 2), plt.imshow(output_np, cmap="gray"), plt.title("Motion Trail (CPU)")
    plt.subplot(1, 3, 3), plt.imshow(output_cp_np, cmap="gray"), plt.title("Motion Trail (GPU)")
    for ax in plt.gcf().axes:
        ax.axis("off")
    plt.show()


# ------------------------------------------------------------------ G-5) Convolution GPU 가속
def convolution2d_cp(image, kernel, bias=0.0, stride=1, padding=0):
    """CuPy 2차원 convolution: 커널 위치는 Python 반복, 출력 배열 전체 연산은 GPU."""
    kernel_height, kernel_width = kernel.shape

    padded_image = cp.pad(
        image,
        ((padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0,
    )

    output_height = (padded_image.shape[0] - kernel_height) // stride + 1
    output_width = (padded_image.shape[1] - kernel_width) // stride + 1

    output = cp.zeros((output_height, output_width), dtype=cp.float32)

    for kernel_y in range(kernel_height):
        for kernel_x in range(kernel_width):
            image_region = padded_image[
                kernel_y:kernel_y + stride * output_height:stride,
                kernel_x:kernel_x + stride * output_width:stride,
            ]
            output += kernel[kernel_y, kernel_x] * image_region

    output += bias

    return output


def run_convolution(input_np):
    # ----- NumPy CPU 처리
    print("[Convolution] CPU 처리 중... (시간이 걸릴 수 있습니다)")
    cpu_start = time.perf_counter()
    output_dx_np = convolution2d(input_np, dx_edge_filter, dx_bias, stride=1, padding=1)
    output_dy_np = convolution2d(input_np, dy_edge_filter, dy_bias, stride=1, padding=1)
    cpu_time = time.perf_counter() - cpu_start

    relu_dx_np = relu(output_dx_np)
    relu_dy_np = relu(output_dy_np)

    # ----- CuPy GPU 처리 (변환 - 워밍업 - 처리)
    input_cp = cp.asarray(input_np)
    dx_filter_cp = cp.asarray(dx_edge_filter)
    dy_filter_cp = cp.asarray(dy_edge_filter)
    sync()

    _ = convolution2d_cp(cp.zeros((64, 64), dtype=cp.float32), dx_filter_cp, dx_bias, stride=1, padding=1)
    sync()

    sync()
    gpu_start = time.perf_counter()
    output_dx_cp = convolution2d_cp(input_cp, dx_filter_cp, dx_bias, stride=1, padding=1)
    output_dy_cp = convolution2d_cp(input_cp, dy_filter_cp, dy_bias, stride=1, padding=1)
    sync()
    gpu_time = time.perf_counter() - gpu_start

    relu_dx_cp_np = cp.asnumpy(cp.maximum(0, output_dx_cp))
    relu_dy_cp_np = cp.asnumpy(cp.maximum(0, output_dy_cp))

    # ----- CPU/GPU 결과 비교 및 출력
    print("========== NumPy CPU ==========")
    print(f"Processing time: {cpu_time:.4f} seconds")
    print()
    print("========== CuPy GPU ==========")
    print(f"Processing time: {gpu_time:.4f} seconds")
    print()
    print("========== Comparison ==========")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")
    print(f"Vertical match: {np.allclose(relu_dx_np, relu_dx_cp_np, atol=1e-5)}")
    print(f"Horizontal match: {np.allclose(relu_dy_np, relu_dy_cp_np, atol=1e-5)}")
    print(f"Max diff (vertical): {np.abs(relu_dx_np - relu_dx_cp_np).max():.2e}")
    print(f"Max diff (horizontal): {np.abs(relu_dy_np - relu_dy_cp_np).max():.2e}")

    plt.figure(figsize=(18, 8))
    plt.subplot(2, 3, 1), plt.imshow(input_np, cmap="gray"), plt.title("Input Image")
    plt.subplot(2, 3, 2), plt.imshow(relu_dx_np, cmap="gray"), plt.title("Vertical Convolution (CPU)")
    plt.subplot(2, 3, 3), plt.imshow(relu_dx_cp_np, cmap="gray"), plt.title("Vertical Convolution (GPU)")
    plt.subplot(2, 3, 4), plt.imshow(input_np, cmap="gray"), plt.title("Input Image")
    plt.subplot(2, 3, 5), plt.imshow(relu_dy_np, cmap="gray"), plt.title("Horizontal Convolution (CPU)")
    plt.subplot(2, 3, 6), plt.imshow(relu_dy_cp_np, cmap="gray"), plt.title("Horizontal Convolution (GPU)")
    for ax in plt.gcf().axes:
        ax.axis("off")
    plt.show()


# ------------------------------------------------------------------ main
def main():
    # F. MNIST CNN 학습 및 시각화 (CPU)
    run_mnist_cnn()

    # G. CuPy GPU 실습
    if not GPU_AVAILABLE:
        print("[알림] CuPy 또는 NVIDIA GPU를 사용할 수 없어 GPU 실습(잔상/Convolution)은 건너뜁니다.")
        return

    input_np = load_input_image()

    plt.imshow(input_np, cmap="gray")
    plt.axis("off")
    plt.show()

    run_motion_trail(input_np)
    run_convolution(input_np)


if __name__ == "__main__":
    main()