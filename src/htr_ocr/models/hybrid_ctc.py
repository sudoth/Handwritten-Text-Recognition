from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_positional_encoding_1d(length: int, dim: int, device: torch.device) -> torch.Tensor:
    pe = torch.zeros(length, dim, device=device)
    position = torch.arange(0, length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # [1, T, D]


class ConvFeatureExtractor(nn.Module):
    """CNN extractor for line images.

    Width downsample factor = 4
    Height is reduced and then collapsed by max over height.
    """

    def __init__(self, out_channels: int = 256, dropout: float = 0.1) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),  # /2

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),  # /4

            nn.Conv2d(128, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Dropout2d(p=float(dropout)),
        )

        self.width_downsample_factor = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)  # [B, C, H', W']
        x = x.max(dim=2).values  # [B, C, W']
        x = x.transpose(1, 2)  # [B, W', C]
        return x


class HybridCTC(nn.Module):
    """CNN -> BiLSTM -> Transformer Encoder -> CTC"""

    def __init__(
        self,
        vocab_size: int,
        cnn_out_channels: int = 256,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        transformer_dim: int = 512,
        transformer_layers: int = 2,
        n_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.vocab_size = int(vocab_size)
        self.transformer_dim = int(transformer_dim)

        self.cnn = ConvFeatureExtractor(
            out_channels=int(cnn_out_channels),
            dropout=float(dropout),
        )

        self.bilstm = nn.LSTM(
            input_size=int(cnn_out_channels),
            hidden_size=int(lstm_hidden),
            num_layers=int(lstm_layers),
            batch_first=True,
            bidirectional=True,
            dropout=float(dropout) if int(lstm_layers) > 1 else 0.0,
        )

        bilstm_out_dim = int(lstm_hidden) * 2
        if bilstm_out_dim != int(transformer_dim):
            self.proj = nn.Linear(bilstm_out_dim, int(transformer_dim))
        else:
            self.proj = nn.Identity()

        enc_layer = nn.TransformerEncoderLayer(
            d_model=int(transformer_dim),
            nhead=int(n_heads),
            dim_feedforward=int(ffn_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=enc_layer,
            num_layers=int(transformer_layers),
        )

        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(int(transformer_dim), self.vocab_size)

        self.time_downsample_factor = self.cnn.width_downsample_factor

    @torch.no_grad()
    def token_lengths_from_widths(self, widths: list[int] | torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(widths):
            widths = torch.tensor(widths, dtype=torch.long)
        return (widths + self.time_downsample_factor - 1) // self.time_downsample_factor

    def forward(
        self,
        x: torch.Tensor,
        token_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feat = self.cnn(x)  # [B, T, C]
        feat, _ = self.bilstm(feat)  # [B, T, 2H]
        feat = self.proj(feat)  # [B, T, D]

        bsz, seq_len, dim = feat.shape

        pos = sinusoidal_positional_encoding_1d(seq_len, dim, feat.device)
        feat = feat + pos
        feat = self.dropout(feat)

        key_padding_mask = None
        if token_lengths is not None:
            token_lengths = token_lengths.to(device=feat.device, dtype=torch.long)
            token_lengths = torch.clamp(token_lengths, max=seq_len)
            ar = torch.arange(seq_len, device=feat.device).unsqueeze(0).expand(bsz, seq_len)
            key_padding_mask = ar >= token_lengths.unsqueeze(1)  # True = padding

        feat = self.transformer(feat, src_key_padding_mask=key_padding_mask)  # [B, T, D]
        logits = self.head(feat)  # [B, T, V]
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.transpose(0, 1)  # [T, B, V]