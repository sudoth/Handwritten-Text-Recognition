from pathlib import Path

import torch
from PIL import Image

from htr_ocr.data.transforms import make_image_transform
from htr_ocr.models.crnn_ctc import CRNNCTC
from htr_ocr.text.ctc_decode import ctc_beam_search_batch, ctc_greedy_decode_batch
from htr_ocr.text.ctc_tokenizer import CTCTokenizer


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[CRNNCTC, CTCTokenizer]:
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    tok = CTCTokenizer.from_dict(ckpt["tokenizer"])
    model = CRNNCTC(
        num_classes=tok.vocab_size,
        in_ch=1,
        rnn_hidden=int(ckpt["cfg"]["model"].get("rnn_hidden", 256)),
        rnn_layers=int(ckpt["cfg"]["model"].get("rnn_layers", 2)),
        fc_hidden=int(ckpt["cfg"]["model"].get("fc_hidden", 256)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tok


@torch.no_grad()
def infer_one(
    checkpoint_path: str | Path,
    image_path: str | Path,
    *,
    height: int = 128,
    keep_aspect: bool = True,
    pad_value: int = 255,
    device_str: str = "cuda",
    decode_method: str = "beam",
    beam_width: int = 50,
    topk: int = 20,
) -> str:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model, tok = load_checkpoint(checkpoint_path, device)

    transform = make_image_transform(
        height=height,
        keep_aspect=keep_aspect,
        tight_crop_enabled=False,
        tight_crop_threshold=245,
        tight_crop_margin=2,
        is_train=False,
        augment_cfg=None,
        fill=int(pad_value),
        to_float_tensor=True,
    )

    img = Image.open(image_path).convert("L")
    x = transform(img)  # [1, H, W] float in [0,1]
    x = x.unsqueeze(0).to(device)  # [B=1,1,H,W]

    log_probs = model(x)  # [T,1,C]

    method = str(decode_method)
    if method == "beam":
        pred = ctc_beam_search_batch(log_probs, tok, beam_width=int(beam_width), topk=int(topk))[0]
    else:
        pred = ctc_greedy_decode_batch(log_probs, tok)[0]

    return pred
