from dataclasses import dataclass
from typing import Callable, Optional

import torch


@dataclass
class SAMCfg:
    enabled: bool = True
    rho: float = 0.05
    adaptive: bool = False


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (SAM).

    base_opt = AdamW(...)
    opt = SAM(model.parameters(), base_optimizer=AdamW, rho=0.05, lr=..., weight_decay=...)
    loss = opt.step(closure)
    """

    def __init__(
        self,
        params,
        base_optimizer: type[torch.optim.Optimizer],
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs,
    ):
        if rho <= 0.0:
            raise ValueError("SAM rho must be > 0")
        self.rho = float(rho)
        self.adaptive = bool(adaptive)

        self.base_optimizer = base_optimizer(params, **kwargs)
        defaults = dict(rho=self.rho, adaptive=self.adaptive, **kwargs)
        super().__init__(self.base_optimizer.param_groups, defaults)

    @torch.no_grad()
    def _grad_norm(self) -> torch.Tensor:
        device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if self.adaptive:
                    g = g * torch.abs(p)
                norms.append(torch.norm(g, p=2))
        if not norms:
            return torch.tensor(0.0, device=device)
        return torch.norm(torch.stack(norms), p=2)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = True) -> None:
        grad_norm = self._grad_norm()
        if grad_norm.item() == 0.0:
            return
        scale = self.rho / (grad_norm + 1e-12)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad
                if self.adaptive:
                    e_w = e_w * torch.abs(p)
                e_w = e_w * scale.to(p.device)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = True) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if "e_w" not in self.state[p]:
                    continue
                p.sub_(self.state[p]["e_w"])

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad(set_to_none=True)

    def step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        if closure is None:
            raise ValueError("SAM requires closure that re-computes loss")

        loss = closure()
        self.first_step(zero_grad=True)
        loss_2 = closure()
        self.second_step(zero_grad=True)
        return loss_2