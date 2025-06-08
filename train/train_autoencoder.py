# train/train_autoencoder.py

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model.autoencoder_lstm import LSTMAutoEncoder

def train_autoencoder(X_tensor, epochs=20, batch_size=32, lr=0.001):
    dataset = TensorDataset(X_tensor, X_tensor)  # 입력 = 출력
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LSTMAutoEncoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for batch_x, _ in loader:
            output = model(batch_x)
            loss = criterion(output, batch_x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[{epoch+1:02d}/{epochs}] Loss: {total_loss / len(loader):.6f}")

    return model
