import math
from dataclasses import dataclass
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from htr_ocr.data.collate import collate_line_batch
from htr_ocr.data.dataset import IamLineDataset
from htr_ocr.data.samplers import BucketBatchSampler
from htr_ocr.data.transforms import make_image_transform
from htr_ocr.models.vt_ctc import HTRVTCTC, SpanMaskCfg
from htr_ocr.optim.sam import SAM
from htr_ocr.text.ctc_tokenizer import CTCTokenizer, build_or_load_vocab
from htr_ocr.text.ctc_decode import ctc_beam_search_batch, ctc_greedy_decode_batch
from htr_ocr.utils.metrics import cer, wer
from htr_ocr.utils.io import ensure_dir
from htr_ocr.utils.repro import seed_everything


@dataclass
class TrainResult:
    best_checkpoint: Path
    best_val_cer: float
    best_val_wer: float


def _ctc_prepare_targets(tokenizer: CTCTokenizer, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    ids = [torch.tensor(tokenizer.encode(t), dtype=torch.long) for t in texts]
    lengths = torch.tensor([len(x) for x in ids], dtype=torch.long)
    targets = torch.cat(ids, dim=0) if ids else torch.empty((0,), dtype=torch.long)
    return targets, lengths


def make_dataloader(cfg, split: str) -> DataLoader:
    processed_dir = Path(cfg.data.processed_dir)
    csv_path = processed_dir / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    is_train = split == "train"

    transform = make_image_transform(
        height=int(cfg.preprocess.height),
        keep_aspect=bool(cfg.preprocess.keep_aspect),
        tight_crop_enabled=bool(cfg.preprocess.tight_crop.enabled),
        tight_crop_threshold=int(cfg.preprocess.tight_crop.threshold),
        tight_crop_margin=int(cfg.preprocess.tight_crop.margin),
        augment_cfg=(cfg.augment if is_train else None),
        is_train=is_train,
        fill=int(cfg.preprocess.pad_value),
        to_float_tensor=True,
    )

    ds = IamLineDataset(csv_path=csv_path, transform=transform, target_height=int(cfg.preprocess.height))

    bucket_enabled = bool(cfg.loader.bucket.enabled) and is_train
    batch_size = int(cfg.loader.batch_size)

    if bucket_enabled:
        lengths = [ds.approx_resized_width(i) for i in range(len(ds))]
        lengths_i = [int(v) for v in lengths if v is not None]
        sampler = BucketBatchSampler(
            lengths=lengths_i,
            batch_size=batch_size,
            shuffle_batches=bool(cfg.loader.shuffle),
            seed=int(cfg.loader.bucket.seed),
            drop_last=bool(cfg.loader.bucket.drop_last),
        )
        dl = DataLoader(
            ds,
            batch_sampler=sampler,
            num_workers=int(cfg.loader.num_workers),
            pin_memory=bool(cfg.loader.pin_memory),
            collate_fn=lambda b: collate_line_batch(b, pad_value=float(cfg.preprocess.pad_value) / 255.0),
        )
        return dl

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=bool(cfg.loader.shuffle) if is_train else False,
        num_workers=int(cfg.loader.num_workers),
        pin_memory=bool(cfg.loader.pin_memory),
        collate_fn=lambda b: collate_line_batch(b, pad_value=float(cfg.preprocess.pad_value) / 255.0),
    )
    return dl


@torch.no_grad()
def evaluate(model: HTRVTCTC, dl: DataLoader, tokenizer: CTCTokenizer, device: torch.device, decode_cfg) -> dict[str, float]:
    model.eval()
    ctc_loss = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)

    total_loss = 0.0
    total_cer = 0.0
    total_wer = 0.0
    n = 0

    for batch in tqdm(dl, desc="eval", leave=False):
        x = batch["pixel_values"].to(device)
        widths = batch["widths"]
        texts = batch["texts"]

        token_lengths = model.token_lengths_from_widths(widths).to(device)

        log_probs = model(x, token_lengths=token_lengths)  # [T,B,V]
        T = int(log_probs.shape[0])
        input_lengths = torch.clamp(token_lengths, max=T)

        targets, target_lengths = _ctc_prepare_targets(tokenizer, texts)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

        method = str(getattr(decode_cfg, "method", "greedy"))
        if method == "beam":
            preds = ctc_beam_search_batch(
                log_probs,
                tokenizer,
                beam_width=int(getattr(decode_cfg, "beam_width", 50)),
                topk=int(getattr(decode_cfg, "topk", 20)),
            )
        else:
            preds = ctc_greedy_decode_batch(log_probs, tokenizer)

        bs = len(texts)
        total_loss += float(loss.item()) * bs
        total_cer += sum(cer(p, t) for p, t in zip(preds, texts))
        total_wer += sum(wer(p, t) for p, t in zip(preds, texts))
        n += bs

    return {
        "loss": total_loss / max(1, n),
        "cer": total_cer / max(1, n),
        "wer": total_wer / max(1, n),
    }


def train_htr_vt_ctc(cfg) -> TrainResult:
    seed_everything(int(cfg.train.seed), deterministic=bool(getattr(cfg.train, "deterministic", True)))
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    tokenizer = build_or_load_vocab(cfg)

    span_cfg = SpanMaskCfg(
        enabled=bool(cfg.span_mask.enabled),
        mask_ratio=float(cfg.span_mask.mask_ratio),
        span_len=int(cfg.span_mask.span_len),
    )

    model = HTRVTCTC(
        vocab_size=tokenizer.vocab_size,
        embed_dim=int(cfg.model.embed_dim),
        n_heads=int(cfg.model.n_heads),
        n_layers=int(cfg.model.n_layers),
        ffn_dim=int(cfg.model.ffn_dim),
        dropout=float(cfg.model.dropout),
        span_mask=span_cfg,
    ).to(device)

    train_dl = make_dataloader(cfg, "train")
    val_dl = make_dataloader(cfg, "val")

    opt_cfg = getattr(cfg.train, "optimizer", None)
    opt_name = str(getattr(opt_cfg, "name", "adamw")).lower()
    lr = float(getattr(opt_cfg, "lr", getattr(cfg.train, "lr", 3e-4)))
    weight_decay = float(getattr(opt_cfg, "weight_decay", getattr(cfg.train, "weight_decay", 1e-5)))
    betas = tuple(getattr(opt_cfg, "betas", [0.9, 0.999]))
    adam_eps = float(getattr(opt_cfg, "eps", 1e-8))

    if len(betas) != 2:
        raise ValueError("train.optimizer.betas must contain exactly two values: [beta1, beta2]")

    if opt_name == "adam":
        base_optimizer = torch.optim.Adam
    elif opt_name == "adamw":
        base_optimizer = torch.optim.AdamW
    else:
        raise ValueError(f"Unsupported train.optimizer.name={opt_name}. Expected one of: adamw, adam")

    use_sam = bool(cfg.train.sam.enabled)
    if use_sam:
        optimizer = SAM(
            model.parameters(),
            base_optimizer=base_optimizer,
            lr=lr,
            weight_decay=weight_decay,
            betas=(float(betas[0]), float(betas[1])),
            eps=adam_eps,
            rho=float(cfg.train.sam.rho),
            adaptive=bool(cfg.train.sam.adaptive),
        )
    else:
        optimizer = base_optimizer(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(float(betas[0]), float(betas[1])),
            eps=adam_eps,
        )

    ctc_loss = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)

    runs_dir = Path(cfg.train.runs_dir)
    run_dir = runs_dir / "htr_vt_ctc"
    ensure_dir(run_dir)

    best_val_cer = float("inf")
    best_val_wer = float("inf")
    best_path = run_dir / "best.pt"
    
    patience = int(cfg.train.patience)
    bad_epochs = 0

    max_epochs = int(cfg.train.max_epochs)

    for epoch in range(1, max_epochs + 1):
        model.train()
        pbar = tqdm(train_dl, desc=f"train epoch {epoch}", leave=False)

        epoch_loss = 0.0
        seen = 0

        for batch in pbar:
            x = batch["pixel_values"].to(device)
            widths = batch["widths"]
            texts = batch["texts"]

            token_lengths = model.token_lengths_from_widths(widths).to(device)

            targets, target_lengths = _ctc_prepare_targets(tokenizer, texts)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            def closure() -> torch.Tensor:
                optimizer.zero_grad(set_to_none=True)
                log_probs = model(x, token_lengths=token_lengths)  # [T,B,V]
                T = int(log_probs.shape[0])
                input_lengths = torch.clamp(token_lengths, max=T)
                loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
                loss.backward()
                return loss

            if use_sam:
                loss = optimizer.step(closure)
            else:
                loss = closure()
                optimizer.step()

            bs = len(texts)
            epoch_loss += float(loss.item()) * bs
            seen += bs
            pbar.set_postfix(loss=float(loss.item()))

        train_loss = epoch_loss / max(1, seen)

        val_metrics = evaluate(model, val_dl, tokenizer, device, decode_cfg=cfg.decode)

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_metrics["loss"], step=epoch)
        mlflow.log_metric("val_cer", val_metrics["cer"], step=epoch)
        mlflow.log_metric("val_wer", val_metrics["wer"], step=epoch)

        improved = val_metrics["cer"] < best_val_cer
        if improved:
            best_val_cer = float(val_metrics["cer"])
            best_val_wer = float(val_metrics["wer"])
            bad_epochs = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "tokenizer": {"id2char": tokenizer.id2char},
                    "cfg": {"model": dict(cfg.model), "preprocess": dict(cfg.preprocess)},
                },
                best_path,
            )
            if bool(getattr(cfg.train, "log_checkpoint_to_mlflow", True)):
                mlflow.log_artifact(str(best_path), artifact_path="checkpoints")
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            break

    return TrainResult(best_checkpoint=best_path, best_val_cer=best_val_cer, best_val_wer=best_val_wer)
