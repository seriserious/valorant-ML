# model/autoencoder_lstm.py

import torch
import torch.nn as nn

class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, input_size, num_layers, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h_repeated = h.repeat(x.size(1), 1, 1).permute(1, 0, 2)  # (batch, seq, feature)
        output, _ = self.decoder(h_repeated)
        return output
