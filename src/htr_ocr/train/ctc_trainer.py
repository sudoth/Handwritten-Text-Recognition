import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import mlflow

from htr_ocr.data.collate import collate_line_batch
from htr_ocr.data.dataset import IamLineDataset
from htr_ocr.data.samplers import BucketBatchSampler
from htr_ocr.data.transforms import make_image_transform
from htr_ocr.models.crnn_ctc import CRNNCTC
from htr_ocr.text.ctc_decode import ctc_beam_search_batch, ctc_greedy_decode_batch
from htr_ocr.text.ctc_tokenizer import CTCTokenizer, build_or_load_vocab
from htr_ocr.utils.metrics import AverageMeter, cer, wer
from htr_ocr.utils.repro import seed_everything


@dataclass
class TrainResult:
    best_checkpoint: Path
    best_val_cer: float
    best_val_wer: float


def _ctc_prepare_targets(tokenizer: CTCTokenizer, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    targets: list[int] = []
    lengths: list[int] = []
    for t in texts:
        ids = tokenizer.encode(t)
        targets.extend(ids)
        lengths.append(len(ids))
    return torch.tensor(targets, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


def _input_lengths_from_widths(widths: list[int], downsample: int) -> torch.Tensor:
    lens = [max(1, int(w) // downsample) for w in widths]
    return torch.tensor(lens, dtype=torch.long)


def _decode_batch(log_probs: torch.Tensor, tokenizer: CTCTokenizer, decode_cfg) -> list[str]:
    method = str(getattr(decode_cfg, "method", "greedy"))
    if method == "beam":
        return ctc_beam_search_batch(
            log_probs,
            tokenizer,
            beam_width=int(getattr(decode_cfg, "beam_width", 50)),
            topk=int(getattr(decode_cfg, "topk", 20)),
        )
    return ctc_greedy_decode_batch(log_probs, tokenizer)


def make_dataloader(
    cfg,
    split_name: str,
) -> DataLoader:
    processed_dir = Path(cfg.data.processed_dir)
    csv_path = processed_dir / f"{split_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    is_train = split_name == "train"
    transform = make_image_transform(
        height=int(cfg.preprocess.height),
        keep_aspect=bool(cfg.preprocess.keep_aspect),
        tight_crop_enabled=bool(cfg.preprocess.tight_crop.enabled),
        tight_crop_threshold=int(cfg.preprocess.tight_crop.threshold),
        tight_crop_margin=int(cfg.preprocess.tight_crop.margin),
        augment_cfg=getattr(cfg, "augment", None) if is_train else None,
        is_train=is_train,
        fill=int(cfg.preprocess.pad_value),
        to_float_tensor=True,
    )

    ds = IamLineDataset(csv_path=csv_path, transform=transform, target_height=int(cfg.preprocess.height))

    bucket_enabled = bool(getattr(cfg.loader.bucket, "enabled", False))
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
        shuffle=bool(cfg.loader.shuffle),
        num_workers=int(cfg.loader.num_workers),
        pin_memory=bool(cfg.loader.pin_memory),
        collate_fn=lambda b: collate_line_batch(b, pad_value=float(cfg.preprocess.pad_value) / 255.0),
    )

def evaluate(
    model: CRNNCTC,
    dl: DataLoader,
    tokenizer: CTCTokenizer,
    device: torch.device,
    decode_cfg,
    blank_id: int = 0,
) -> dict[str, float]:
    model.eval()
    ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    loss_m = AverageMeter()
    cer_m = AverageMeter()
    wer_m = AverageMeter()

    for batch in tqdm(dl, desc="eval", leave=False):
        x = batch["pixel_values"].to(device)
        widths = batch["widths"]
        texts = batch["texts"]

        with torch.no_grad():
            log_probs = model(x)  # [T,B,C]
            input_lengths = _input_lengths_from_widths(widths, model.time_downsample_factor).to(device)
            targets, target_lengths = _ctc_prepare_targets(tokenizer, texts)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

        preds = _decode_batch(log_probs, tokenizer, decode_cfg)

        loss_m.update(float(loss.item()), n=len(texts))
        for p, t in zip(preds, texts, strict=False):
            cer_m.update(cer(p, t))
            wer_m.update(wer(p, t))

    return {"loss": loss_m.avg, "cer": cer_m.avg, "wer": wer_m.avg}


def train_crnn_ctc(cfg) -> TrainResult:
    seed_everything(int(cfg.train.seed), deterministic=bool(getattr(cfg.train, "deterministic", True)))

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    tokenizer = build_or_load_vocab(cfg)

    train_dl = make_dataloader(cfg, "train")
    val_dl = make_dataloader(cfg, "val")

    model = CRNNCTC(
        num_classes=tokenizer.vocab_size,
        in_ch=1,
        rnn_hidden=int(cfg.model.rnn_hidden),
        rnn_layers=int(cfg.model.rnn_layers),
        fc_hidden=int(cfg.model.fc_hidden),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
        betas=(float(cfg.train.adam_beta1), float(cfg.train.adam_beta2)),
        eps=float(cfg.train.adam_eps),
    )

    ctc_loss = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)

    runs_dir = Path(cfg.train.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    best_val_cer = math.inf
    best_val_wer = math.inf
    best_path = runs_dir / "best.pt"

    patience = int(cfg.train.early_stop.patience)
    bad_epochs = 0

    max_epochs = int(cfg.train.epochs)

    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_m = AverageMeter()

        pbar = tqdm(train_dl, desc=f"train epoch {epoch}", leave=False)
        for batch in pbar:
            x = batch["pixel_values"].to(device)
            widths = batch["widths"]
            texts = batch["texts"]

            log_probs = model(x)  # [T,B,C]
            input_lengths = _input_lengths_from_widths(widths, model.time_downsample_factor).to(device)

            targets, target_lengths = _ctc_prepare_targets(tokenizer, texts)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(cfg.train.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip))
            optimizer.step()

            loss_m.update(float(loss.item()), n=len(texts))
            pbar.set_postfix(loss=f"{loss_m.avg:.4f}")

        val_metrics = evaluate(model, val_dl, tokenizer, device, decode_cfg=cfg.decode)

        mlflow.log_metric("train_loss", loss_m.avg, step=epoch)
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
