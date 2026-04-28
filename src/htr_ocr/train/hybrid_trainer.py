from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from htr_ocr.data.collate import collate_line_batch
from htr_ocr.data.dataset import IamLineDataset
from htr_ocr.data.samplers import BucketBatchSampler
from htr_ocr.data.transforms import make_image_transform
from htr_ocr.models.hybrid_ctc import HybridCTC
from htr_ocr.text.ctc_decode import decode_batch
from htr_ocr.text.ctc_tokenizer import CTCTokenizer, build_or_load_vocab_ctc
from htr_ocr.utils.io import ensure_dir
from htr_ocr.utils.metrics import cer, wer


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


def _build_transform(cfg, is_train: bool):
    return make_image_transform(
        height=int(cfg.preprocess.height),
        keep_aspect=bool(cfg.preprocess.keep_aspect),
        tight_crop_enabled=bool(cfg.preprocess.tight_crop.enabled),
        tight_crop_threshold=int(cfg.preprocess.tight_crop.threshold),
        tight_crop_margin=int(cfg.preprocess.tight_crop.margin),
        augment_cfg=(cfg.augment if is_train and bool(cfg.augment.enabled) else None),
        is_train=is_train,
        fill=int(cfg.preprocess.pad_value),
        to_float_tensor=True,
    )


def make_dataloader(cfg, split: str) -> DataLoader:
    processed_dir = Path(cfg.data.processed_dir)
    csv_path = processed_dir / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    is_train = split == "train"
    transform = _build_transform(cfg, is_train=is_train)

    ds = IamLineDataset(
        csv_path=csv_path,
        transform=transform,
        target_height=int(cfg.preprocess.height),
    )

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
        return DataLoader(
            ds,
            batch_sampler=sampler,
            num_workers=int(cfg.loader.num_workers),
            pin_memory=bool(cfg.loader.pin_memory),
            collate_fn=lambda b: collate_line_batch(b, pad_value=float(cfg.preprocess.pad_value) / 255.0),
        )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=bool(cfg.loader.shuffle) if is_train else False,
        num_workers=int(cfg.loader.num_workers),
        pin_memory=bool(cfg.loader.pin_memory),
        collate_fn=lambda b: collate_line_batch(b, pad_value=float(cfg.preprocess.pad_value) / 255.0),
    )


def _make_scheduler(optimizer, cfg, total_steps: int):
    name = str(cfg.train.scheduler.name).lower()
    if name in {"none", ""}:
        return None

    warmup_ratio = float(cfg.train.scheduler.warmup_ratio)
    warmup_steps = int(total_steps * warmup_ratio)

    def _lr_lambda(current_step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)

        if name == "linear":
            return max(0.0, 1.0 - progress)

        if name == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        raise ValueError(f"Unknown scheduler: {name}")

    return LambdaLR(optimizer, lr_lambda=_lr_lambda)


@torch.inference_mode()
def evaluate(model: HybridCTC, dl: DataLoader, tokenizer: CTCTokenizer, device: torch.device, decode_cfg) -> dict[str, float]:
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

        log_probs = model(x, token_lengths=token_lengths)  # [T, B, V]
        t_steps = int(log_probs.shape[0])
        input_lengths = torch.clamp(token_lengths, max=t_steps)

        targets, target_lengths = _ctc_prepare_targets(tokenizer, texts)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

        preds = decode_batch(
            log_probs=log_probs,
            tokenizer=tokenizer,
            method=str(getattr(decode_cfg, "method", "greedy")),
            beam_width=int(getattr(decode_cfg, "beam_width", 50)),
            topk=int(getattr(decode_cfg, "topk", 20)),
        )

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


def train_hybrid_ctc(cfg) -> TrainResult:
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    tokenizer = build_or_load_vocab_ctc(cfg)

    model = HybridCTC(
        vocab_size=tokenizer.vocab_size,
        cnn_out_channels=int(cfg.model.cnn_out_channels),
        lstm_hidden=int(cfg.model.lstm_hidden),
        lstm_layers=int(cfg.model.lstm_layers),
        transformer_dim=int(cfg.model.transformer_dim),
        transformer_layers=int(cfg.model.transformer_layers),
        n_heads=int(cfg.model.n_heads),
        ffn_dim=int(cfg.model.ffn_dim),
        dropout=float(cfg.model.dropout),
    ).to(device)

    train_dl = make_dataloader(cfg, "train")
    val_dl = make_dataloader(cfg, "val")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.optimizer.lr),
        weight_decay=float(cfg.train.optimizer.weight_decay),
        betas=(float(cfg.train.optimizer.betas[0]), float(cfg.train.optimizer.betas[1])),
        eps=float(cfg.train.optimizer.eps),
    )

    grad_accum_steps = int(cfg.train.grad_accum_steps)
    total_steps = int(cfg.train.max_epochs) * max(1, (len(train_dl) + grad_accum_steps - 1) // grad_accum_steps)
    scheduler = _make_scheduler(optimizer, cfg, total_steps)

    ctc_loss = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)

    use_amp = bool(cfg.train.amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    run_dir = Path(cfg.train.runs_dir) / "hybrid_ctc"
    ensure_dir(run_dir)
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"

    model_cfg = {
        "cnn_out_channels": int(cfg.model.cnn_out_channels),
        "lstm_hidden": int(cfg.model.lstm_hidden),
        "lstm_layers": int(cfg.model.lstm_layers),
        "transformer_dim": int(cfg.model.transformer_dim),
        "transformer_layers": int(cfg.model.transformer_layers),
        "n_heads": int(cfg.model.n_heads),
        "ffn_dim": int(cfg.model.ffn_dim),
        "dropout": float(cfg.model.dropout),
    }

    best_val_cer = float("inf")
    best_val_wer = float("inf")
    bad_epochs = 0

    for epoch in range(1, int(cfg.train.max_epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_loss = 0.0
        seen = 0

        pbar = tqdm(train_dl, desc=f"train e{epoch}", leave=False)
        for step, batch in enumerate(pbar, start=1):
            x = batch["pixel_values"].to(device)
            widths = batch["widths"]
            texts = batch["texts"]

            token_lengths = model.token_lengths_from_widths(widths).to(device)

            targets, target_lengths = _ctc_prepare_targets(tokenizer, texts)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                log_probs = model(x, token_lengths=token_lengths)
                t_steps = int(log_probs.shape[0])
                input_lengths = torch.clamp(token_lengths, max=t_steps)
                loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
                loss_for_backward = loss / grad_accum_steps

            scaler.scale(loss_for_backward).backward()

            if step % grad_accum_steps == 0 or step == len(train_dl):
                if float(cfg.train.max_grad_norm) > 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), float(cfg.train.max_grad_norm))

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                if scheduler is not None:
                    scheduler.step()

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
        mlflow.log_metric("lr", float(optimizer.param_groups[0]["lr"]), step=epoch)

        torch.save(
            {
                "model": model.state_dict(),
                "tokenizer": tokenizer.to_dict(),
                "model_cfg": model_cfg,
            },
            last_path,
        )

        improved = val_metrics["cer"] < best_val_cer
        if improved:
            best_val_cer = val_metrics["cer"]
            best_val_wer = val_metrics["wer"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "tokenizer": tokenizer.to_dict(),
                    "model_cfg": model_cfg,
                },
                best_path,
            )
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= int(cfg.train.patience):
            break

    return TrainResult(
        best_checkpoint=best_path,
        best_val_cer=best_val_cer,
        best_val_wer=best_val_wer,
    )