# preprocess/marker_extractor.py

import cv2
import numpy as np
import torch
import os

def prepare_all_data(data_dir="data", window_size=30):
    all_sequences = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".mp4"):
            video_path = os.path.join(data_dir, filename)
            print(f"🔍 Processing: {filename}")
            tensor = prepare_data(video_path, window_size)
            if tensor.size(0) > 0:  # 시퀀스가 존재하면
                all_sequences.append(tensor)

    if not all_sequences:
        raise ValueError("No valid sequences found in any video.")

    return torch.cat(all_sequences, dim=0)  # shape: (total_sequences, 30, 2)

# 🔧 마커 좌표 추출
def extract_marker_coordinates(video_path):
    cap = cv2.VideoCapture(video_path)
    coords = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

        lower_red2 = np.array([160, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        mask = mask1 | mask2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            cx, cy = x + w // 2, y + h // 2
            coords.append([cx, cy])
        else:
            coords.append([0, 0])

    cap.release()
    return np.array(coords)

# 📏 정규화
def normalize_coords(coords, width=1920, height=1080):
    return coords / np.array([[width, height]])

# 🔁 시퀀스 생성
def create_sequences(data, window_size=30):
    sequences = []
    for i in range(len(data) - window_size):
        seq = data[i:i+window_size]
        sequences.append(seq)
    return np.array(sequences)

# ❌ 0으로만 된 시퀀스 제거
def filter_zero_sequences(sequences):
    return np.array([seq for seq in sequences if not np.all(seq == 0)])

# 📦 전체 전처리 파이프라인 (X만 반환)
def prepare_data(video_path, window_size=30):
    coords = extract_marker_coordinates(video_path)
    coords = normalize_coords(coords)
    sequences = create_sequences(coords, window_size)
    sequences = filter_zero_sequences(sequences)
    return torch.tensor(sequences, dtype=torch.float32)
