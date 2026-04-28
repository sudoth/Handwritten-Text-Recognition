from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from PIL import Image

from htr_ocr.data.transforms import make_image_transform
from htr_ocr.models.hybrid_ctc import HybridCTC
from htr_ocr.text.ctc_decode import decode_batch
from htr_ocr.text.ctc_tokenizer import CTCTokenizer


def load_checkpoint(path: Path, device: torch.device) -> Tuple[HybridCTC, CTCTokenizer]:
    ckpt = torch.load(path, map_location=device)

    tok = CTCTokenizer.from_dict(ckpt["tokenizer"])
    model_cfg = ckpt["model_cfg"]

    model = HybridCTC(
        vocab_size=tok.vocab_size,
        cnn_out_channels=int(model_cfg["cnn_out_channels"]),
        lstm_hidden=int(model_cfg["lstm_hidden"]),
        lstm_layers=int(model_cfg["lstm_layers"]),
        transformer_dim=int(model_cfg["transformer_dim"]),
        transformer_layers=int(model_cfg["transformer_layers"]),
        n_heads=int(model_cfg["n_heads"]),
        ffn_dim=int(model_cfg["ffn_dim"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)

    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, tok


@torch.inference_mode()
def infer_one(
    checkpoint_path: Path,
    image_path: Path,
    height: int,
    keep_aspect: bool,
    pad_value: int,
    device_str: str,
    decode_method: str = "beam",
    beam_width: int = 50,
    topk: int = 20,
) -> str:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model, tok = load_checkpoint(checkpoint_path, device)

    tf = make_image_transform(
        height=int(height),
        keep_aspect=bool(keep_aspect),
        tight_crop_enabled=False,
        tight_crop_threshold=0,
        tight_crop_margin=0,
        augment_cfg=None,
        is_train=False,
        fill=int(pad_value),
        to_float_tensor=True,
    )

    img = Image.open(image_path).convert("L")
    x = tf(img).unsqueeze(0).to(device)  # [1, 1, H, W]

    widths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
    token_lengths = model.token_lengths_from_widths(widths).to(device)

    log_probs = model(x, token_lengths=token_lengths)  # [T, 1, V]
    pred = decode_batch(
        log_probs=log_probs,
        tokenizer=tok,
        method=str(decode_method),
        beam_width=int(beam_width),
        topk=int(topk),
    )[0]
    return pred