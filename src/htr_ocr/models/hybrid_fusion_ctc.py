from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from htr_ocr.models.hybrid_ctc import ConvFeatureExtractor, sinusoidal_positional_encoding_1d


class HybridFusionCTC(nn.Module):
    """CNN -> BiLSTM -> Transformer Encoder -> Fusion(local, global) -> CTC.

    Отличие от HybridCTC:
    - HybridCTC использует только выход Transformer Encoder;
    - HybridFusionCTC объединяет выход BiLSTM и выход Transformer Encoder на каждом timestep.
    """

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
        fusion_type: str = "concat",
    ) -> None:
        super().__init__()

        self.vocab_size = int(vocab_size)
        self.transformer_dim = int(transformer_dim)
        self.fusion_type = str(fusion_type).lower()

        if self.fusion_type not in {"concat", "gated"}:
            raise ValueError(
                f"Unknown fusion_type={fusion_type!r}. Expected one of: concat, gated."
            )

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

        if self.fusion_type == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(int(transformer_dim) * 2, int(transformer_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            )
        else:
            self.gate = nn.Linear(int(transformer_dim) * 2, int(transformer_dim))
            self.fusion_dropout = nn.Dropout(float(dropout))

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

        local_feat = feat

        batch_size, seq_len, dim = feat.shape

        pos = sinusoidal_positional_encoding_1d(seq_len, dim, feat.device)
        feat = feat + pos
        feat = self.dropout(feat)

        key_padding_mask = None
        if token_lengths is not None:
            token_lengths = token_lengths.to(device=feat.device, dtype=torch.long)
            token_lengths = torch.clamp(token_lengths, max=seq_len)
            ar = torch.arange(seq_len, device=feat.device).unsqueeze(0).expand(batch_size, seq_len)
            key_padding_mask = ar >= token_lengths.unsqueeze(1)

        global_feat = self.transformer(
            feat,
            src_key_padding_mask=key_padding_mask,
        )  # [B, T, D]

        mixed = torch.cat([local_feat, global_feat], dim=-1)  # [B, T, 2D]

        if self.fusion_type == "concat":
            feat = self.fusion(mixed)  # [B, T, D]
        else:
            gate = torch.sigmoid(self.gate(mixed))  # [B, T, D]
            feat = gate * global_feat + (1.0 - gate) * local_feat
            feat = self.fusion_dropout(feat)

        logits = self.head(feat)  # [B, T, V]
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs.transpose(0, 1)  # [T, B, V]