# model/autoencoder_lstm.py

import torch
import torch.nn as nn

class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, latent_dim=32, seq_len=30):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.latent = nn.Linear(hidden_dim, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)
        self.output_layer = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # Encoder
        enc_out, (h_n, _) = self.encoder(x)  # (batch, seq_len, hidden_dim)
        h_last = h_n[-1]                     # 마지막 layer의 hidden state
        z = self.latent(h_last)              # 잠재 벡터 (batch, latent_dim)

        # Decoder 초기 입력 반복
        dec_input = self.decoder_input(z).unsqueeze(1).repeat(1, x.size(1), 1)

        # Decoder
        dec_out, _ = self.decoder(dec_input)
        output = self.output_layer(dec_out)  # (batch, seq_len, input_dim)
        return output
