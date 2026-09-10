from __future__ import annotations

import numpy as np

from . import const
from .model import HAS_TORCH, Circuit, RandomModel
from .task import make_batch

if HAS_TORCH:
    import torch
    import torch.nn.functional as F

    from .model import TinyLM, TorchRunner


def train_tiny(steps: int = const.MAX_TRAIN_STEPS, seed: int = const.SEED):
    if not HAS_TORCH:
        return None, "torch unavailable"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = TinyLM()
    opt = torch.optim.AdamW(net.parameters(), lr=const.LR)
    net.train()
    last_acc = 0.0
    for step in range(steps):
        tokens, gold = make_batch(const.BATCH, rng)
        xt = torch.as_tensor(tokens, dtype=torch.long)
        yt = torch.as_tensor(gold, dtype=torch.long)
        logits = net(xt)
        loss = F.cross_entropy(logits, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            last_acc = 100.0 * float((logits.argmax(-1) == yt).float().mean())
            if last_acc >= 99.0 and step >= 100:
                break
    net.eval()
    return TorchRunner(net), f"trained {step + 1} steps acc={last_acc:.1f}"


def untrained_runner():
    return RandomModel(seed=1)


def circuit_runner():
    return Circuit()
