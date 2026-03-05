from pathlib import Path

import fire
import mlflow
import pandas as pd
import torch
from rich.console import Console
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
from PIL import Image
import torch.nn.functional as F

from htr_ocr.config_loader import load_cfg
from htr_ocr.data.collate import collate_line_batch
from htr_ocr.data.dataset import IamLineDataset
from htr_ocr.data.iam import build_manifest
from htr_ocr.data.samplers import BucketBatchSampler
from htr_ocr.data.splits import make_group_split
from htr_ocr.data.transforms import make_image_transform
from htr_ocr.train.ctc_infer import infer_one, load_checkpoint
from htr_ocr.train.ctc_trainer import evaluate, make_dataloader, train_crnn_ctc
from htr_ocr.train.vt_trainer import evaluate as vt_evaluate, make_dataloader as vt_make_dataloader, train_htr_vt_ctc
from htr_ocr.train.vt_infer import infer_one as vt_infer_one, load_checkpoint as vt_load_checkpoint
from htr_ocr.utils.io import ensure_dir
from htr_ocr.utils.mlflow_utils import mlflow_run

console = Console()


class HTRCLI:
    def make_manifest(self, *overrides: str) -> None:
        cfg = load_cfg("make_manifest", overrides=list(overrides))

        ensure_dir(cfg.data.processed_dir)

        with mlflow_run("make_manifest", cfg):
            df = build_manifest(
                images_root=cfg.data.images_root,
                annotations_path=cfg.data.annotations_path,
                forms_path=getattr(cfg.data, "forms_path", None),
                keep_status=list(cfg.data.keep_status),
                limit=int(cfg.data.limit),
            )

            manifest_path = Path(cfg.data.manifest_path)
            df.to_parquet(manifest_path, index=False)
            console.print(f"Saved manifest: {manifest_path} (rows={len(df)})")

    def make_splits(self, *overrides: str) -> None:
        cfg = load_cfg("make_splits", overrides=list(overrides))

        processed_dir = Path(cfg.data.processed_dir)
        manifest_path = Path(cfg.data.manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {manifest_path}.")

        ensure_dir(processed_dir)

        with mlflow_run("make_splits", cfg):
            df = pd.read_parquet(manifest_path)

            strategy = str(cfg.split.strategy)
            if strategy == "writer":
                group_col = "writer_id"
            elif strategy == "form":
                group_col = "form_id"
            else:
                raise ValueError(f"Unknown split.strategy={strategy}")

            train_df, val_df, test_df = make_group_split(
                df,
                group_col=group_col,
                seed=int(cfg.split.seed),
                train=float(cfg.split.train),
                val=float(cfg.split.val),
                test=float(cfg.split.test),
            )

            (processed_dir / "train.csv").write_text(train_df.to_csv(index=False), encoding="utf-8")
            (processed_dir / "val.csv").write_text(val_df.to_csv(index=False), encoding="utf-8")
            (processed_dir / "test.csv").write_text(test_df.to_csv(index=False), encoding="utf-8")

            console.print(
                "Saved splits to data/processed: "
                f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)} (group={group_col})"
            )

    def inspect_data(self, *overrides: str) -> None:
        cfg = load_cfg("inspect_data", overrides=list(overrides))

        processed_dir = Path(cfg.data.processed_dir)
        split_name = str(cfg.loader.split)
        csv_path = processed_dir / f"{split_name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")

        transform = make_image_transform(
            height=int(cfg.preprocess.height),
            keep_aspect=bool(cfg.preprocess.keep_aspect),
            tight_crop_enabled=bool(cfg.preprocess.tight_crop.enabled),
            tight_crop_threshold=int(cfg.preprocess.tight_crop.threshold),
            tight_crop_margin=int(cfg.preprocess.tight_crop.margin),
            augment_cfg=None,
            is_train=False,
            fill=int(cfg.preprocess.pad_value),
            to_float_tensor=True,
        )

        ds = IamLineDataset(csv_path=csv_path, transform=transform, target_height=int(cfg.preprocess.height))

        bucket_enabled = bool(cfg.loader.bucket.enabled)
        batch_size = int(cfg.loader.batch_size)

        if bucket_enabled:
            lengths = [ds.approx_resized_width(i) for i in range(len(ds))]
            if any(v is None for v in lengths):
                console.print("No requested columns")
                bucket_enabled = False
            else:
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

        if not bucket_enabled:
            dl = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=bool(cfg.loader.shuffle),
                num_workers=int(cfg.loader.num_workers),
                pin_memory=bool(cfg.loader.pin_memory),
                collate_fn=lambda b: collate_line_batch(b, pad_value=float(cfg.preprocess.pad_value) / 255.0),
            )

        with mlflow_run("inspect_data", cfg, extra_tags={"split": split_name}):
            n_batches = int(cfg.loader.n_batches)
            console.print(
                f"Inspecting split={split_name} batches={n_batches} bs={batch_size} "
                f"height={int(cfg.preprocess.height)} bucket={bucket_enabled}"
            )

            for bi, batch in enumerate(dl):
                if bi >= n_batches:
                    break
                pv = batch["pixel_values"]
                widths = batch["widths"]
                w_max = int(pv.shape[-1])

                grid = make_grid(pv.detach().cpu(), nrow=2, padding=2)
                ensure_dir(cfg.loader.samples_path)
                to_pil_image(grid).save(Path(cfg.loader.samples_path) / f"grid_{bi}.png")

                console.print(
                    f"batch[{bi}]: pixel_values={tuple(pv.shape)} w_max={w_max} "
                    f"min_w={min(widths)} max_w={max(widths)}"
                )

                t0 = batch["texts"][0]
                console.print(f"  sample text[0]: {t0[:120]}")

    def inspect_augmentations(self, image_path: str | None = None, *overrides: str) -> None:
        cfg = load_cfg("inspect_augmentations", overrides=list(overrides))

        processed_dir = Path(cfg.data.processed_dir)
        split_name = str(cfg.loader.split)

        if image_path is None:
            csv_path = processed_dir / f"{split_name}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Split CSV not found: {csv_path}")
            df = pd.read_csv(csv_path)
            idx = int(cfg.inspect_aug.index)
            idx = max(0, min(idx, len(df) - 1))
            image_path = str(df.iloc[idx]["image_path"])

        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at {img_path}")

        base_tf = make_image_transform(
            height=int(cfg.preprocess.height),
            keep_aspect=bool(cfg.preprocess.keep_aspect),
            tight_crop_enabled=bool(cfg.preprocess.tight_crop.enabled),
            tight_crop_threshold=int(cfg.preprocess.tight_crop.threshold),
            tight_crop_margin=int(cfg.preprocess.tight_crop.margin),
            augment_cfg=None,
            is_train=False,
            fill=int(cfg.preprocess.pad_value),
            to_float_tensor=True,
        )

        aug_tf = make_image_transform(
            height=int(cfg.preprocess.height),
            keep_aspect=bool(cfg.preprocess.keep_aspect),
            tight_crop_enabled=bool(cfg.preprocess.tight_crop.enabled),
            tight_crop_threshold=int(cfg.preprocess.tight_crop.threshold),
            tight_crop_margin=int(cfg.preprocess.tight_crop.margin),
            augment_cfg=cfg.augment,
            is_train=True,
            fill=int(cfg.preprocess.pad_value),
            to_float_tensor=True,
        )

        img = Image.open(img_path).convert("L")

        base = base_tf(img)  # [1,H,W]
        n = int(cfg.inspect_aug.n)
        cols = int(cfg.inspect_aug.cols)
        samples = [base] + [aug_tf(img) for _ in range(n)]

        pad_value = float(cfg.preprocess.pad_value) / 255.0
        max_w = max(int(t.shape[-1]) for t in samples)
        padded: list[torch.Tensor] = []
        for t in samples:
            w = int(t.shape[-1])
            if w < max_w:
                t = F.pad(t, (0, max_w - w, 0, 0), value=pad_value)
            padded.append(t)

        batch = torch.stack(padded, dim=0)  # [N,1,H,W]
        grid = make_grid(batch.detach().cpu(), nrow=max(1, cols), padding=2)

        ensure_dir(cfg.inspect_aug.out_dir)
        out_dir = Path(cfg.inspect_aug.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "aug_grid.png"
        to_pil_image(grid).save(out_path)

        with mlflow_run("inspect_augmentations", cfg, extra_tags={"split": split_name}):
            mlflow.log_artifact(str(out_path), artifact_path="augmentations")

        console.print(f"Saved augmentation grid: {out_path}")

    def train_crnn_ctc(self, *overrides: str) -> None:
        cfg = load_cfg("train_crnn_ctc", overrides=list(overrides))

        with mlflow_run("train_crnn_ctc", cfg):
            result = train_crnn_ctc(cfg)

            device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
            model, tok = load_checkpoint(result.best_checkpoint, device)
            test_dl = make_dataloader(cfg, "test")
            metrics = evaluate(model, test_dl, tok, device, decode_cfg=cfg.decode)

            mlflow.log_metric("test_loss", metrics["loss"])
            mlflow.log_metric("test_cer", metrics["cer"])
            mlflow.log_metric("test_wer", metrics["wer"])

            console.print(
                f"Best checkpoint={result.best_checkpoint} "
                f"val_CER={result.best_val_cer:.4f} val_WER={result.best_val_wer:.4f} "
                f"test_CER={metrics['cer']:.4f} test_WER={metrics['wer']:.4f}"
            )

    def eval_crnn_ctc(self, *overrides: str) -> None:
        cfg = load_cfg("eval_crnn_ctc", overrides=list(overrides))

        split_name = str(cfg.eval.split)
        ckpt_path = Path(cfg.eval.checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

        with mlflow_run("eval_crnn_ctc", cfg, extra_tags={"split": split_name}):
            device = torch.device(cfg.eval.device if torch.cuda.is_available() else "cpu")
            model, tok = load_checkpoint(ckpt_path, device)
            dl = make_dataloader(cfg, split_name)
            metrics = evaluate(model, dl, tok, device, decode_cfg=cfg.decode)

            mlflow.log_metric(f"{split_name}_loss", metrics["loss"])
            mlflow.log_metric(f"{split_name}_cer", metrics["cer"])
            mlflow.log_metric(f"{split_name}_wer", metrics["wer"])

            console.print(
                f"split={split_name}: loss={metrics['loss']:.4f} "
                f"CER={metrics['cer']:.4f} WER={metrics['wer']:.4f}"
            )

    def infer_crnn_ctc(self, *overrides: str) -> None:
        cfg = load_cfg("infer_crnn_ctc", overrides=list(overrides))

        ckpt_path = Path(cfg.infer.checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

        image_path = Path(cfg.infer.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        pred = infer_one(
            checkpoint_path=ckpt_path,
            image_path=image_path,
            height=int(cfg.preprocess.height),
            keep_aspect=bool(cfg.preprocess.keep_aspect),
            pad_value=int(cfg.preprocess.pad_value),
            device_str=str(cfg.infer.device),
            decode_method=str(getattr(cfg.decode, "method", "greedy")),
            beam_width=int(getattr(cfg.decode, "beam_width", 50)),
            topk=int(getattr(cfg.decode, "topk", 20)),
        )
        console.print(f"{pred}")


def main() -> None:
    fire.Fire(HTRCLI)