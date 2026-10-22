"""Logit calibration used by Joint IE score lattices."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


class Calibrator(ABC):
    """Interface for transformations applied to logits before sigmoid."""

    @abstractmethod
    def calibrate(self, logits: Any) -> Any:
        """Return calibrated logits, preserving scalar/container shape."""

    def __call__(self, logits: Any) -> Any:
        return self.calibrate(logits)


@dataclass(frozen=True)
class IdentityCalibrator(Calibrator):
    def calibrate(self, logits: Any) -> Any:
        return logits


@dataclass(frozen=True)
class TemperatureCalibrator(Calibrator):
    """Divide logits by a positive temperature."""

    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be greater than zero")

    def calibrate(self, logits: Any) -> Any:
        try:
            return logits / self.temperature
        except TypeError:
            if isinstance(logits, tuple):
                return tuple(self.calibrate(value) for value in logits)
            if isinstance(logits, list):
                return [self.calibrate(value) for value in logits]
            if isinstance(logits, dict):
                return {key: self.calibrate(value) for key, value in logits.items()}
            raise


def fit_binary_temperature(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    max_iter: int = 50,
) -> float:
    """Fit one positive temperature by minimizing binary dev-set NLL."""
    logits = logits.detach().float()
    targets = targets.detach().float()
    log_temperature = torch.zeros((), requires_grad=True, device=logits.device)
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(1e-3, 1e3)
        loss = F.binary_cross_entropy_with_logits(logits / temperature, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(1e-3, 1e3).cpu())


def expected_calibration_error(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    bins: int = 15,
) -> float:
    """Compute equal-width binary expected calibration error."""
    probabilities = probabilities.detach().float().reshape(-1)
    targets = targets.detach().float().reshape(-1)
    edges = torch.linspace(0, 1, bins + 1, device=probabilities.device)
    error = probabilities.new_zeros(())
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        keep = (probabilities >= lower) & (
            probabilities <= upper if index == bins - 1 else probabilities < upper
        )
        if keep.any():
            error = error + keep.float().mean() * (
                probabilities[keep].mean() - targets[keep].mean()
            ).abs()
    return float(error.cpu())
