import torch


def sample_span_mask(
    lengths: torch.Tensor,
    mask_ratio: float,
    span_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Span mask для последовательностей

    lengths: [B] длины токенов без паддинга
    Return: mask [B, T_max]  True заменены на mask_token
    Паддинг не учитывается в маске.
    """
    lengths = lengths.to(device=device, dtype=torch.long)
    B = int(lengths.shape[0])
    T_max = int(lengths.max().item()) if B > 0 else 0
    if T_max == 0:
        return torch.zeros((B, 0), dtype=torch.bool, device=device)

    span_len = max(1, int(span_len))
    mask = torch.zeros((B, T_max), dtype=torch.bool, device=device)

    for i in range(B):
        L = int(lengths[i].item())
        if L <= 0:
            continue
        n_to_mask = int(round(float(mask_ratio) * L))
        n_to_mask = max(0, min(n_to_mask, L))
        if n_to_mask == 0:
            continue

        # итеративная генерация спанов одинаковой длины, as described in appendix-like text. :contentReference[oaicite:6]{index=6}
        masked = 0
        max_start = max(0, L - span_len)
        # чтобы не было бесконечного цикла
        for _ in range(10_000):
            if masked >= n_to_mask:
                break
            start = int(torch.randint(low=0, high=max_start + 1, size=(1,), device=device).item())
            end = min(L, start + span_len)
            newly = (~mask[i, start:end]).sum().item()
            mask[i, start:end] = True
            masked += int(newly)

        # ensure we don't accidentally mask padding (positions >= L)
        if L < T_max:
            mask[i, L:] = False

    return mask