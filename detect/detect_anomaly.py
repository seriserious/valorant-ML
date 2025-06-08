import torch
import torch.nn as nn
from preprocess.marker_extractor import prepare_data
from model.autoencoder_lstm import LSTMAutoEncoder
import matplotlib.pyplot as plt


def detect_anomaly(video_path, model_path="trained_model.pth", threshold=None):
    print(f"📼 Analyzing {video_path}")
    
    # 1. 데이터 전처리
    X = prepare_data(video_path, window_size=30)
    if len(X) == 0:
        print("⚠️ 유효한 시퀀스가 없습니다.")
        return

    # 2. 모델 불러오기
    model = LSTMAutoEncoder()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # 3. 복원 및 오차 계산
    criterion = nn.MSELoss(reduction='none')
    with torch.no_grad():
        reconstructed = model(X)
        loss = criterion(reconstructed, X)  # shape: (N, 30, 2)
        loss = loss.mean(dim=(1, 2))        # shape: (N,) → 시퀀스별 MSE

    # 4. 임계값 설정 (없으면 자동 계산)
    if threshold is None:
        # 평균 + 3 * 표준편차로 이상치 기준 설정
        threshold = loss.mean().item() + 3 * loss.std().item()

    print(f"📊 Threshold: {threshold:.6f}")
    
    # 5. 이상 시퀀스 판단
    anomalies = (loss > threshold).numpy()
    for i, is_anomaly in enumerate(anomalies):
        status = "🚨 이상" if is_anomaly else "✅ 정상"
        print(f"[{i:03d}] MSE: {loss[i]:.6f} → {status}")

    # 6. 시각화
    plt.figure(figsize=(10, 4))
    plt.plot(loss.numpy(), label="Reconstruction Error")
    plt.axhline(threshold, color='red', linestyle='--', label="Threshold")
    plt.title("Sequence-wise Reconstruction Error")
    plt.xlabel("Sequence Index")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.show()
