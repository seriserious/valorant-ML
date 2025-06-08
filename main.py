from preprocess.marker_extractor import prepare_all_data
from train.train_autoencoder import train_autoencoder
from detect.detect_anomaly import detect_anomaly
import torch

# 1. 데이터 준비
X_tensor = prepare_all_data("data", window_size=30)

# 2. 모델 학습
model = train_autoencoder(X_tensor)

# 3. 모델 저장
torch.save(model.state_dict(), "trained_model.pth")

# 4. 이상 감지 수행
detect_anomaly("data/clip_007.mp4", model_path="trained_model.pth")
