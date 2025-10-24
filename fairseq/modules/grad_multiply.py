"""Реализация GradMultiply, совместимая с патчами WebUI."""

from __future__ import annotations

import torch


class _GradMultiplyFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, scale: float):  # type: ignore[override]
        ctx.scale = scale
        return tensor

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        return grad_output * ctx.scale, None


class GradMultiply:
    """Объект с методом forward, чтобы WebUI мог его переписать."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, scale: float):  # pragma: no cover - обёртка
        return _GradMultiplyFn.apply(tensor, scale)

    @staticmethod
    def apply(tensor: torch.Tensor, scale: float):  # pragma: no cover - совместимость
        return _GradMultiplyFn.apply(tensor, scale)


def grad_multiply(tensor: torch.Tensor, scale: float):
    return _GradMultiplyFn.apply(tensor, scale)


__all__ = ["GradMultiply", "grad_multiply"]
