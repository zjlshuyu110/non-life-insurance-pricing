"""An ordinary fully connected MLP with an actuarially appropriate loss."""
from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.metrics import mean_tweedie_deviance
from torch import nn


class PricingMLP(nn.Module):
    """Two ReLU hidden layers; positive annual-rate output through a log link."""

    def __init__(self, inputs: int, widths: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        for width in widths:
            layers.extend([nn.Linear(inputs, width), nn.ReLU()])
            inputs = width
        output = nn.Linear(inputs, 1)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # A numerical guard on the log-relative rate; not a cap on observed losses.
        return self.layers(x).squeeze(-1).clamp(-8, 8)


def tweedie_objective(log_relative_rate, relative_target, normalized_exposure, power=1.9):
    """Half deviance up to a target-only constant, on a fixed scaled target.

    Exposure is normalized using the entire fitting sample, not each minibatch.
    Thus uniform minibatches give unbiased gradients of the weighted objective.
    """
    return (normalized_exposure * (
        torch.exp((2 - power) * log_relative_rate) / (2 - power)
        - relative_target * torch.exp((1 - power) * log_relative_rate) / (1 - power)
    )).mean()


def predict(model: PricingMLP, x: np.ndarray, scale: float) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(x), 32768):
            log_rate = model(torch.from_numpy(x[start:start + 32768]))
            chunks.append(scale * np.exp(log_rate.numpy().astype(np.float64)))
    return np.concatenate(chunks)


def fit_mlp(x, target, exposure, widths, seed, config, *, validation=None, epochs=None):
    """Validation selects stopping epoch; final refits use that fixed epoch budget."""
    started = time.perf_counter()
    torch.manual_seed(seed)
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    model = PricingMLP(x.shape[1], widths)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    scale = float(np.dot(target, exposure) / exposure.sum())
    if not scale > 0:
        raise ValueError("Training losses must have a positive mean.")
    tx = torch.from_numpy(np.asarray(x, dtype=np.float32))
    ty = torch.tensor(target / scale, dtype=torch.float32)
    tw = torch.tensor(exposure / exposure.mean(), dtype=torch.float32)
    rng = np.random.default_rng(seed)
    history, best, best_epoch, stale = [], float("inf"), 0, 0
    for epoch in range(1, (epochs or config["max_epochs"]) + 1):
        model.train()
        order = rng.permutation(len(x))
        for start in range(0, len(x), config["batch_size"]):
            ids = order[start:start + config["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = tweedie_objective(model(tx[ids]), ty[ids], tw[ids], config["tweedie_power"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite neural loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()
        train_pred = predict(model, x, scale)
        record = {"epoch": epoch, "train_deviance": mean_tweedie_deviance(
            target, train_pred, power=config["tweedie_power"], sample_weight=exposure)}
        if validation is not None:
            vx, vy, vw = validation
            record["validation_deviance"] = mean_tweedie_deviance(
                vy, predict(model, vx, scale), power=config["tweedie_power"], sample_weight=vw)
            if record["validation_deviance"] < best - config["min_delta"]:
                best, best_epoch, stale = record["validation_deviance"], epoch, 0
            else:
                stale += 1
        history.append(record)
        if epoch == 1 or epoch % 5 == 0:
            suffix = f", validation={record['validation_deviance']:.5f}" if validation is not None else ""
            print(f"MLP {widths}, seed {seed}, epoch {epoch}: train={record['train_deviance']:.5f}{suffix}", flush=True)
        if validation is not None and stale >= config["patience"]:
            break
    return model, scale, history, {
        "widths": widths, "seed": seed, "best_epoch": best_epoch if validation is not None else epoch,
        "best_validation_deviance": best if validation is not None else None,
        "epochs_run": epoch, "seconds": round(time.perf_counter() - started, 3),
        "parameters": sum(p.numel() for p in model.parameters()),
    }
