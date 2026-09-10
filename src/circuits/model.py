"""Tiny causal transformer (torch) plus a compiled copy circuit (numpy)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import const

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    HAS_TORCH = False


def _as_set(heads) -> set[tuple[int, int]]:
    if not heads:
        return set()
    return {(int(l), int(h)) for l, h in heads}


@dataclass
class Circuit:
    """Compiled induction copy: layer-0 head-0 writes token-at-pos-1 into the last position."""

    vocab: int = const.VOCAB
    n_layers: int = 1
    n_heads: int = 2

    def collect(self, tokens: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
        eye = np.eye(self.vocab, dtype=np.float32)
        copy_z = eye[tokens[:, 1]]
        zeros = np.zeros((len(tokens), self.vocab), dtype=np.float32)
        acts = {(0, 0): copy_z}
        for h in range(1, self.n_heads):
            acts[(0, h)] = zeros
        return acts

    def logits(
        self,
        tokens: np.ndarray,
        ablate=(),
        patch: dict[tuple[int, int], np.ndarray] | None = None,
    ) -> np.ndarray:
        ablate_set = _as_set(ablate)
        acts = self.collect(tokens)
        if patch:
            acts.update(patch)
        out = np.zeros((len(tokens), self.vocab), dtype=np.float32)
        for head, z in acts.items():
            if head not in ablate_set:
                out = out + z
        return out


class RandomModel:
    """Untrained: logits ignore the copy structure."""

    def __init__(self, seed: int = 1):
        self.rng = np.random.default_rng(seed)
        self.n_layers = 1
        self.n_heads = 2
        self._w = self.rng.normal(0, 1, size=(const.VOCAB, const.VOCAB)).astype(np.float32)

    def collect(self, tokens: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
        b = len(tokens)
        noise = self.rng.normal(0, 0.1, size=(b, const.VOCAB)).astype(np.float32)
        return {(0, 0): noise, (0, 1): np.zeros((b, const.VOCAB), dtype=np.float32)}

    def logits(self, tokens: np.ndarray, ablate=(), patch=None) -> np.ndarray:
        # Hash-free: map last token through a fixed random matrix. Copy token does not win.
        last = tokens[:, -1]
        out = self._w[last]
        return out


if HAS_TORCH:

    class Block(nn.Module):
        def __init__(self, d_model: int, n_heads: int, d_ff: int = 0):
            super().__init__()
            assert d_model % n_heads == 0
            self.n_heads = n_heads
            self.dh = d_model // n_heads
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
            self.o = nn.Linear(d_model, d_model, bias=False)
            self.ln1 = nn.LayerNorm(d_model)
            self.ff = None
            if d_ff:
                self.ln2 = nn.LayerNorm(d_model)
                self.ff = nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Linear(d_ff, d_model),
                )

        def attn(self, x, ablate_heads: set[int], patch: dict[int, torch.Tensor] | None):
            b, t, d = x.shape
            qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.dh)
            q, k, v = qkv.unbind(2)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            scale = self.dh ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            causal = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), 1)
            scores = scores.masked_fill(causal, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            z = torch.matmul(attn, v)
            for h in ablate_heads:
                z[:, h] = 0
            if patch:
                for h, val in patch.items():
                    z[:, h] = val
            merged = z.transpose(1, 2).contiguous().view(b, t, d)
            return self.o(merged), z

        def forward(self, x, ablate_heads: set[int], patch: dict[int, torch.Tensor] | None):
            a, z = self.attn(self.ln1(x), ablate_heads, patch)
            x = x + a
            if self.ff is not None:
                x = x + self.ff(self.ln2(x))
            return x, z

    class TinyLM(nn.Module):
        def __init__(
            self,
            vocab: int = const.VOCAB,
            n_layers: int = const.N_LAYERS,
            n_heads: int = const.N_HEADS,
            d_model: int = const.D_MODEL,
            d_ff: int = const.D_FF,
            seq_len: int = const.SEQ_LEN,
        ):
            super().__init__()
            self.n_layers = n_layers
            self.n_heads = n_heads
            self.tok = nn.Embedding(vocab, d_model)
            self.pos = nn.Embedding(seq_len, d_model)
            self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
            self.ln = nn.LayerNorm(d_model)
            self.unembed = nn.Linear(d_model, vocab, bias=False)

        def _run(self, tokens: torch.Tensor, ablate, patch, want_z: bool):
            b, t = tokens.shape
            x = self.tok(tokens) + self.pos(torch.arange(t, device=tokens.device))
            ablate_set = _as_set(ablate)
            z_all = {}
            for li, block in enumerate(self.blocks):
                heads = {h for l, h in ablate_set if l == li}
                p = None
                if patch:
                    p = {h: v for (l, h), v in patch.items() if l == li}
                x, z = block(x, heads, p)
                if want_z:
                    for h in range(self.n_heads):
                        z_all[(li, h)] = z[:, h]
            logits = self.unembed(self.ln(x))
            return logits, z_all

        def forward(self, tokens: torch.Tensor, ablate=(), patch=None):
            logits, _ = self._run(tokens, ablate, patch, want_z=False)
            return logits[:, -1]

        def collect_torch(self, tokens: torch.Tensor):
            _, z = self._run(tokens, (), None, want_z=True)
            return z

    class TorchRunner:
        def __init__(self, net: TinyLM, device: str = "cpu"):
            self.net = net.to(device).eval()
            self.device = device
            self.n_layers = net.n_layers
            self.n_heads = net.n_heads

        def _t(self, tokens: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(tokens, device=self.device, dtype=torch.long)

        def collect(self, tokens: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
            with torch.no_grad():
                z = self.net.collect_torch(self._t(tokens))
            return {k: v.detach().cpu().numpy() for k, v in z.items()}

        def logits(self, tokens: np.ndarray, ablate=(), patch=None) -> np.ndarray:
            p = None
            if patch:
                p = {
                    k: torch.as_tensor(v, device=self.device, dtype=torch.float32)
                    for k, v in patch.items()
                }
            with torch.no_grad():
                out = self.net(self._t(tokens), ablate=ablate, patch=p)
            return out.detach().cpu().numpy()

else:
    TinyLM = None  # type: ignore
    TorchRunner = None  # type: ignore
