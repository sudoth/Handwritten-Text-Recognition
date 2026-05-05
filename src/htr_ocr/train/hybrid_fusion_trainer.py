from dataclasses import dataclass
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from htr_ocr.models.hybrid_fusion_ctc import HybridFusionCTC
from htr_ocr.text.ctc_tokenizer import build_or_load_vocab
from htr_ocr.train.hybrid_trainer import (
    _ctc_prepare_targets,
    _make_scheduler,
    evaluate,
    make_dataloader,
)
from htr_ocr.utils.io import ensure_dir


@dataclass
class TrainResult:
    best_checkpoint: Path
    best_val_cer: float
    best_val_wer: float


def _build_checkpoint_payload(model: HybridFusionCTC, tokenizer, cfg) -> dict:
    return {
        "model_state": model.state_dict(),
        "tokenizer": {
            "id2char": tokenizer.id2char,
        },
        "cfg": {
            "model": OmegaConf.to_container(cfg.model, resolve=True),
            "preprocess": OmegaConf.to_container(cfg.preprocess, resolve=True),
        },
    }


def train_hybrid_fusion_ctc(cfg) -> TrainResult:
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    tokenizer = build_or_load_vocab(cfg)

    model = HybridFusionCTC(
        vocab_size=tokenizer.vocab_size,
        cnn_out_channels=int(cfg.model.cnn_out_channels),
        lstm_hidden=int(cfg.model.lstm_hidden),
        lstm_layers=int(cfg.model.lstm_layers),
        transformer_dim=int(cfg.model.transformer_dim),
        transformer_layers=int(cfg.model.transformer_layers),
        n_heads=int(cfg.model.n_heads),
        ffn_dim=int(cfg.model.ffn_dim),
        dropout=float(cfg.model.dropout),
        fusion_type=str(cfg.model.fusion_type),
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
    steps_per_epoch = max(
        1,
        (len(train_dl) + grad_accum_steps - 1) // grad_accum_steps,
    )
    total_steps = int(cfg.train.max_epochs) * steps_per_epoch

    scheduler = _make_scheduler(
        optimizer=optimizer,
        cfg=cfg,
        total_steps=total_steps,
        steps_per_epoch=steps_per_epoch,
    )

    ctc_loss = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)

    use_amp = bool(cfg.train.amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    run_dir = Path(cfg.train.runs_dir) / "hybrid_fusion_ctc"
    ensure_dir(run_dir)

    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"

    best_val_cer = float("inf")
    best_val_wer = float("inf")
    bad_epochs = 0

    log_ckpt_to_mlflow = bool(getattr(cfg.train, "log_checkpoint_to_mlflow", True))
    log_last_ckpt_to_mlflow = bool(getattr(cfg.train, "log_last_checkpoint_to_mlflow", False))

    for epoch in range(1, int(cfg.train.max_epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_loss = 0.0
        seen = 0

        pbar = tqdm(train_dl, desc=f"train fusion e{epoch}", leave=False)
        for step, batch in enumerate(pbar, start=1):
            pixel_values = batch["pixel_values"].to(device)
            widths = batch["widths"]
            texts = batch["texts"]

            token_lengths = model.token_lengths_from_widths(widths).to(device)

            targets, target_lengths = _ctc_prepare_targets(tokenizer, texts)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                log_probs = model(pixel_values, token_lengths=token_lengths)
                time_steps = int(log_probs.shape[0])
                input_lengths = torch.clamp(token_lengths, max=time_steps)

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

            batch_size = len(texts)
            epoch_loss += float(loss.item()) * batch_size
            seen += batch_size
            pbar.set_postfix(loss=float(loss.item()))

        train_loss = epoch_loss / max(1, seen)
        val_metrics = evaluate(model, val_dl, tokenizer, device, decode_cfg=cfg.decode)

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_metrics["loss"], step=epoch)
        mlflow.log_metric("val_cer", val_metrics["cer"], step=epoch)
        mlflow.log_metric("val_wer", val_metrics["wer"], step=epoch)
        mlflow.log_metric("lr", float(optimizer.param_groups[0]["lr"]), step=epoch)

        last_payload = _build_checkpoint_payload(model, tokenizer, cfg)
        torch.save(last_payload, last_path)

        if log_last_ckpt_to_mlflow:
            mlflow.log_artifact(str(last_path), artifact_path="checkpoints")

        improved = val_metrics["cer"] < best_val_cer
        if improved:
            best_val_cer = float(val_metrics["cer"])
            best_val_wer = float(val_metrics["wer"])
            bad_epochs = 0

            best_payload = _build_checkpoint_payload(model, tokenizer, cfg)
            torch.save(best_payload, best_path)

            if log_ckpt_to_mlflow:
                mlflow.log_artifact(str(best_path), artifact_path="checkpoints")
        else:
            bad_epochs += 1

        if bad_epochs >= int(cfg.train.patience):
            break

    return TrainResult(
        best_checkpoint=best_path,
        best_val_cer=best_val_cer,
        best_val_wer=best_val_wer,
    )
