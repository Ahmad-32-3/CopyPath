from __future__ import annotations

import numpy as np

from . import const


def make_batch(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Induction copy: [x, y, z, x] with gold y (token after the first x)."""
    x = rng.integers(0, const.VOCAB, size=n)
    y = rng.integers(0, const.VOCAB, size=n)
    z = rng.integers(0, const.VOCAB, size=n)
    tokens = np.stack([x, y, z, x], axis=1).astype(np.int64)
    return tokens, y.astype(np.int64)


def make_corrupt(tokens: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Change the copied token (position 1) so a working copy head predicts the wrong y."""
    corrupt = np.array(tokens, copy=True)
    y = tokens[:, 1]
    shift = 1 + rng.integers(0, const.VOCAB - 1, size=len(y))
    corrupt[:, 1] = (y + shift) % const.VOCAB
    return corrupt


def chance_pct() -> float:
    return 100.0 / const.VOCAB
