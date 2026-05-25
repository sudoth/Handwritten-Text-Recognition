from pathlib import Path
from typing import Tuple

import torch
from PIL import Image

from htr_ocr.data.transforms import make_image_transform
from htr_ocr.models.hybrid_ctc import HybridCTC
from htr_ocr.text.ctc_decode import ctc_beam_search_batch, ctc_greedy_decode_batch
from htr_ocr.text.ctc_tokenizer import CTCTokenizer


def load_checkpoint(path: Path, device: torch.device) -> Tuple[HybridCTC, CTCTokenizer]:
    # Keep current checkpoint loading semantics and avoid FutureWarning about upcoming default change.
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if "cfg" in ckpt and "model" in ckpt["cfg"]:
        model_cfg = ckpt["cfg"]["model"]
    elif "model_cfg" in ckpt:
        model_cfg = ckpt["model_cfg"]
    else:
        raise KeyError("Checkpoint does not contain model config")

    tok_payload = ckpt["tokenizer"]
    tok = CTCTokenizer.from_dict(tok_payload)

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

    state_key = "model_state" if "model_state" in ckpt else "model"
    model.load_state_dict(ckpt[state_key], strict=True)
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
    x = tf(img).unsqueeze(0).to(device)

    widths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
    token_lengths = model.token_lengths_from_widths(widths).to(device)

    log_probs = model(x, token_lengths=token_lengths)

    method = str(decode_method).lower()
    if method == "greedy":
        pred = ctc_greedy_decode_batch(log_probs, tok)[0]
    elif method == "beam":
        pred = ctc_beam_search_batch(
            log_probs,
            tok,
            beam_width=int(beam_width),
            topk=int(topk),
        )[0]
    else:
        raise ValueError(f"Unknown decode method: {method}")

    return pred
