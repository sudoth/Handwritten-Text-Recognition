import math
from dataclasses import dataclass
from typing import Iterable

import torch

from htr_ocr.text.ctc_tokenizer import CTCTokenizer


def _log_add_exp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def ctc_greedy_decode_batch(log_probs: torch.Tensor, tokenizer: CTCTokenizer) -> list[str]:
    """log_probs: [T, B, C]"""
    with torch.no_grad():
        ids = log_probs.argmax(dim=-1)  # [T, B]
        ids = ids.cpu().numpy()

    preds: list[str] = []
    blank = tokenizer.blank_id
    for b in range(ids.shape[1]):
        seq = ids[:, b].tolist()
        collapsed: list[int] = []
        prev = None
        for i in seq:
            if i == prev:
                continue
            prev = i
            if i == blank:
                continue
            collapsed.append(i)
        preds.append(tokenizer.decode_greedy(collapsed))
    return preds


def ctc_beam_search_decode(
    log_probs_tc: torch.Tensor,
    tokenizer: CTCTokenizer,
    *,
    beam_width: int = 50,
    topk: int = 20,
) -> str:
    """log_probs_tc: [T, C]"""

    blank = tokenizer.blank_id

    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, -math.inf)}

    T, C = log_probs_tc.shape
    k = max(1, min(int(topk), int(C)))

    for t in range(int(T)):
        lp = log_probs_tc[t]  # [C]
        vals, idxs = torch.topk(lp, k=k)

        next_beams: dict[tuple[int, ...], tuple[float, float]] = {}

        for prefix, (p_b, p_nb) in beams.items():
            p_total = _log_add_exp(p_b, p_nb)

            for logp, c in zip(vals.tolist(), idxs.tolist(), strict=False):
                c = int(c)

                if c == blank:
                    nb = next_beams.get(prefix, (-math.inf, -math.inf))
                    new_p_b = _log_add_exp(nb[0], p_total + logp)
                    next_beams[prefix] = (new_p_b, nb[1])
                    continue

                end = prefix[-1] if prefix else None
                new_prefix = prefix + (c,)

                if end == c:
                    nb_new = next_beams.get(new_prefix, (-math.inf, -math.inf))
                    new_p_nb = _log_add_exp(nb_new[1], p_b + logp)
                    next_beams[new_prefix] = (nb_new[0], new_p_nb)

                    nb_same = next_beams.get(prefix, (-math.inf, -math.inf))
                    new_p_nb_same = _log_add_exp(nb_same[1], p_nb + logp)
                    next_beams[prefix] = (nb_same[0], new_p_nb_same)
                else:
                    nb_new = next_beams.get(new_prefix, (-math.inf, -math.inf))
                    new_p_nb = _log_add_exp(nb_new[1], p_total + logp)
                    next_beams[new_prefix] = (nb_new[0], new_p_nb)

        beams = dict(
            sorted(
                next_beams.items(),
                key=lambda kv: _log_add_exp(kv[1][0], kv[1][1]),
                reverse=True,
            )[: max(1, int(beam_width))]
        )

    best_prefix = max(beams.items(), key=lambda kv: _log_add_exp(kv[1][0], kv[1][1]))[0]
    return tokenizer.decode_greedy(list(best_prefix))


def ctc_beam_search_batch(
    log_probs: torch.Tensor,
    tokenizer: CTCTokenizer,
    *,
    beam_width: int = 50,
    topk: int = 20,
) -> list[str]:
    """log_probs: [T, B, C]"""
    T, B, C = log_probs.shape
    preds: list[str] = []
    for b in range(int(B)):
        preds.append(
            ctc_beam_search_decode(
                log_probs[:, b, :],
                tokenizer,
                beam_width=beam_width,
                topk=topk,
            )
        )
    return preds


def decode_batch(
    log_probs: torch.Tensor,
    tokenizer: CTCTokenizer,
    *,
    method: str = "greedy",
    beam_width: int = 50,
    topk: int = 20,
) -> list[str]:
    """Universal batch decoder for CTC models.

    Args:
        log_probs: Tensor [T, B, C]
        tokenizer: CTC tokenizer
        method: 'greedy' or 'beam'
        beam_width: beam width for beam search
        topk: top-k candidates per timestep for beam search

    Returns:
        list[str]: decoded strings for batch
    """
    method = str(method).lower().strip()

    if method in {"greedy"}:
        return ctc_greedy_decode_batch(log_probs, tokenizer)

    if method in {"beam", "beem"}:
        return ctc_beam_search_batch(
            log_probs,
            tokenizer,
            beam_width=int(beam_width),
            topk=int(topk),
        )

    raise ValueError(f"Unknown decode method: {method}")