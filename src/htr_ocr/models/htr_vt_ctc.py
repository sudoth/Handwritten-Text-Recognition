import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

from htr_ocr.regularization.span_mask import sample_span_mask


def sinusoidal_positional_encoding_1d(length: int, dim: int, device: torch.device) -> torch.Tensor:
    pe = torch.zeros(length, dim, device=device)
    position = torch.arange(0, length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


class ResNet18LineExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        m = resnet18(weights=None)

        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)

        self.layer1 = m.layer1

        # урезаем только длину, но не высоту:
        self.layer2 = m.layer2
        self._set_block_stride_hw(self.layer2[0], stride_h=2, stride_w=1)

        self.layer3 = m.layer3
        self._set_block_stride_hw(self.layer3[0], stride_h=2, stride_w=1)

        # убираем последний блок

    @staticmethod
    def _set_block_stride_hw(block: nn.Module, stride_h: int, stride_w: int) -> None:
        if hasattr(block, "conv1"):
            block.conv1.stride = (stride_h, stride_w)
        if hasattr(block, "downsample") and block.downsample is not None:
            if isinstance(block.downsample, nn.Sequential) and hasattr(block.downsample[0], "stride"):
                block.downsample[0].stride = (stride_h, stride_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,1,H,W] а реснет хочкт 3 канала, повторяем
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x = self.stem(x)    # /4,/4
        x = self.layer1(x)  # так же
        x = self.layer2(x)  # /2 по H, W так же
        x = self.layer3(x)  # /2 по H, W так же

        # схлопываем токены
        x = x.max(dim=2, keepdim=True).values
        return x  # [B,256,1,W']


@dataclass
class SpanMaskCfg:
    enabled: bool = True
    mask_ratio: float = 0.4
    span_len: int = 8


class HTRVTCTC(nn.Module):
    """CNN -> Transformer Encoder -> CTC

    log_probs: [T,B,C]
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 768,
        n_heads: int = 6,
        n_layers: int = 4,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
        span_mask: Optional[SpanMaskCfg] = None,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)

        self.extractor = ResNet18LineExtractor()
        self.proj = nn.Conv2d(256, self.embed_dim, kernel_size=1)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=int(n_heads),
            dim_feedforward=int(ffn_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(n_layers))

        self.head = nn.Linear(self.embed_dim, self.vocab_size)

        self._mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.normal_(self._mask_token, mean=0.0, std=0.02)

        self.span_mask = span_mask or SpanMaskCfg()

        # ширина уменьшится примерно в 4 раза (conv1+maxpool)
        self.time_downsample_factor = 4

    @torch.no_grad()
    def token_lengths_from_widths(self, widths: list[int] | torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(widths):
            widths = torch.tensor(widths, dtype=torch.long)
        # ceil(width / 4)
        return (widths + self.time_downsample_factor - 1) // self.time_downsample_factor

    def forward(
        self,
        x: torch.Tensor,  # [B,1,H,Wpad]
        token_lengths: Optional[torch.Tensor] = None,  # [B]
    ) -> torch.Tensor:
        feat = self.extractor(x)          # [B,256,1,W']
        feat = self.proj(feat)            # [B,D,1,W']
        feat = feat.squeeze(2).transpose(1, 2)  # [B,W',D]

        B, T, D = feat.shape

        # маска True == padding (PyTorch convention)
        if token_lengths is not None:
            token_lengths = token_lengths.to(device=feat.device, dtype=torch.long)
            token_lengths = torch.clamp(token_lengths, max=T)
            ar = torch.arange(T, device=feat.device).unsqueeze(0).expand(B, T)
            key_padding_mask = ar >= token_lengths.unsqueeze(1)  # [B,T]
        else:
            key_padding_mask = None

        # positional encoding
        pos = sinusoidal_positional_encoding_1d(T, D, feat.device)
        feat = feat + pos

        # span mask (только на трейне)
        if self.training and self.span_mask.enabled and token_lengths is not None:
            mask = sample_span_mask(
                lengths=token_lengths,
                mask_ratio=float(self.span_mask.mask_ratio),
                span_len=int(self.span_mask.span_len),
                device=feat.device,
            )  # [B,T] True только где маска (паддинг никогда не в маске)
            feat = torch.where(mask.unsqueeze(-1), self._mask_token.expand(B, T, D), feat)

        out = self.encoder(feat, src_key_padding_mask=key_padding_mask)  # [B,T,D]
        logits = self.head(out)  # [B,T,V]
        log_probs = F.log_softmax(logits, dim=-1)  # [B,T,V]
        return log_probs.transpose(0, 1)  # [T,B,V]
