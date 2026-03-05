from dataclasses import dataclass
from typing import Any, Callable

from PIL import Image

import torch
from torchvision.transforms.functional import pil_to_tensor

from htr_ocr.data.augmentations import (
    RandomDistort,
    RandomElastic,
    RandomOneOf,
    RandomPerspective,
    RandomRotate,
    RandomShear,
    RandomStretch,
)


@dataclass(frozen=True)
class ResizeToHeight:
    height: int = 128
    keep_aspect: bool = True

    def __call__(self, img: Image.Image) -> Image.Image:
        h = int(self.height)
        if h <= 0:
            raise ValueError("height must be > 0")

        img = img.convert("L")
        w0, h0 = img.size
        if h0 <= 0:
            return img

        if self.keep_aspect:
            w = max(1, int(round(w0 * (h / h0))))
            return img.resize((w, h), resample=Image.Resampling.BILINEAR)
        return img.resize((w0, h), resample=Image.Resampling.BILINEAR)


@dataclass(frozen=True)
class TightCrop:
    enabled: bool = False
    threshold: int = 245
    margin: int = 2

    def __call__(self, img: Image.Image) -> Image.Image:
        if not self.enabled:
            return img

        img = img.convert("L")
        bw = img.point(lambda p: 255 if p < self.threshold else 0, mode="L")
        bbox = bw.getbbox()
        if bbox is None:
            return img

        left, upper, right, lower = bbox
        m = int(self.margin)
        left = max(0, left - m)
        upper = max(0, upper - m)
        right = min(img.size[0], right + m)
        lower = min(img.size[1], lower + m)
        return img.crop((left, upper, right, lower))


def _build_train_augment(augment_cfg: Any, *, fill: int = 255) -> Callable[[Image.Image], Image.Image] | None:
    if augment_cfg is None:
        return None
    if not bool(getattr(augment_cfg, "enabled", False)):
        return None

    methods: list[Callable[[Image.Image], Image.Image]] = []
    p_each = float(getattr(augment_cfg, "p_each", 1.0))

    shear_cfg = getattr(augment_cfg, "shear", None)
    if shear_cfg is not None and bool(getattr(shear_cfg, "enabled", True)):
        methods.append(
            RandomShear(
                max_degrees=float(getattr(shear_cfg, "max_degrees", 7.0)),
                p=p_each,
                fill=fill,
            )
        )

    rot_cfg = getattr(augment_cfg, "rotate", None)
    if rot_cfg is not None and bool(getattr(rot_cfg, "enabled", True)):
        methods.append(
            RandomRotate(
                max_degrees=float(getattr(rot_cfg, "max_degrees", 3.0)),
                p=p_each,
                fill=fill,
            )
        )

    el_cfg = getattr(augment_cfg, "elastic", None)
    if el_cfg is not None and bool(getattr(el_cfg, "enabled", True)):
        methods.append(
            RandomElastic(
                alpha=float(getattr(el_cfg, "alpha", 40.0)),
                sigma=float(getattr(el_cfg, "sigma", 6.0)),
                p=p_each,
                fill=fill,
            )
        )

    geo_cfg = getattr(augment_cfg, "geometric", None)
    if geo_cfg is not None and bool(getattr(geo_cfg, "enabled", True)):
        geo_methods: list[Callable[[Image.Image], Image.Image]] = []

        d_cfg = getattr(geo_cfg, "distortion", None)
        if d_cfg is not None and bool(getattr(d_cfg, "enabled", True)):
            geo_methods.append(
                RandomDistort(
                    max_shift_px=int(getattr(d_cfg, "max_shift_px", 6)),
                    num_stripes=int(getattr(d_cfg, "num_stripes", 12)),
                    p=p_each,
                    fill=fill,
                )
            )

        s_cfg = getattr(geo_cfg, "stretch", None)
        if s_cfg is not None and bool(getattr(s_cfg, "enabled", True)):
            geo_methods.append(
                RandomStretch(
                    max_factor=float(getattr(s_cfg, "max_factor", 0.15)),
                    p=p_each,
                )
            )

        p_cfg = getattr(geo_cfg, "perspective", None)
        if p_cfg is not None and bool(getattr(p_cfg, "enabled", True)):
            geo_methods.append(
                RandomPerspective(
                    distortion_scale=float(getattr(p_cfg, "distortion_scale", 0.2)),
                    p=p_each,
                    fill=fill,
                )
            )

        if geo_methods:
            methods.append(RandomOneOf(transforms=geo_methods, p_total=1.0))

    if not methods:
        return None

    p_total = float(getattr(augment_cfg, "p", 0.5))
    one_of = bool(getattr(augment_cfg, "one_of", True))

    if one_of:
        return RandomOneOf(transforms=methods, p_total=p_total)

    def _apply_all(img: Image.Image) -> Image.Image:
        import random

        if random.random() > p_total:
            return img
        for m in methods:
            img = m(img)
        return img

    return _apply_all


def make_image_transform(
    *,
    height: int = 128,
    keep_aspect: bool = True,
    tight_crop_enabled: bool = False,
    tight_crop_threshold: int = 245,
    tight_crop_margin: int = 2,
    augment_cfg: Any | None = None,
    is_train: bool = False,
    to_float_tensor: bool = True,
    fill: int = 255,
) -> Callable[[Image.Image], Any]:
    
    crop = TightCrop(
        enabled=bool(tight_crop_enabled),
        threshold=int(tight_crop_threshold),
        margin=int(tight_crop_margin),
    )
    resize = ResizeToHeight(height=int(height), keep_aspect=bool(keep_aspect))

    aug = _build_train_augment(augment_cfg, fill=int(fill)) if is_train else None

    def _pil_only(img: Image.Image) -> Image.Image:
        img = img.convert("L")
        img = crop(img)
        if aug is not None:
            img = aug(img)
        img = resize(img)
        return img

    if not to_float_tensor:
        return _pil_only

    def _to_tensor(img: Image.Image):
        img = _pil_only(img)
        t = pil_to_tensor(img)  # uint8, [1,H,W]
        return t.to(dtype=torch.float32) / 255.0

    return _to_tensor
