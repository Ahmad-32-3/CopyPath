from __future__ import annotations

import numpy as np

from . import const
from .task import chance_pct, make_batch, make_corrupt


def accuracy(runner, tokens: np.ndarray, gold: np.ndarray, ablate=(), patch=None) -> float:
    logits = runner.logits(tokens, ablate=ablate, patch=patch)
    pred = logits.argmax(axis=-1)
    return 100.0 * float((pred == gold).mean())


def recover_pct(runner, tokens: np.ndarray, gold: np.ndarray, heads, rng: np.random.Generator) -> float:
    """Patch named heads from clean into a corrupted copy; fraction of restored gold labels."""
    corrupt = make_corrupt(tokens, rng)
    clean_acts = runner.collect(tokens)
    patch = {h: clean_acts[h] for h in heads if h in clean_acts}
    return accuracy(runner, corrupt, gold, patch=patch)


def ablate_pct(runner, tokens: np.ndarray, gold: np.ndarray, heads) -> float:
    return accuracy(runner, tokens, gold, ablate=heads)


def all_heads(n_layers: int, n_heads: int) -> list[tuple[int, int]]:
    return [(l, h) for l in range(n_layers) for h in range(n_heads)]


def sweep_ablation(runner, tokens: np.ndarray, gold: np.ndarray) -> list[dict]:
    base = accuracy(runner, tokens, gold)
    rows = []
    for head in all_heads(runner.n_layers, runner.n_heads):
        pct = ablate_pct(runner, tokens, gold, [head])
        rows.append(
            {
                "layer": head[0],
                "head": head[1],
                "ablate_pct": round(pct, 4),
                "drop": round(base - pct, 4),
            }
        )
    rows.sort(key=lambda r: -r["drop"])
    return rows


def breaking_set(runner, tokens: np.ndarray, gold: np.ndarray, rows: list[dict]) -> list[tuple[int, int]]:
    """Smallest prefix of the sweep whose joint ablation drops task_pct by BREAK_DROP."""
    base = accuracy(runner, tokens, gold)
    chosen: list[tuple[int, int]] = []
    for r in rows:
        chosen.append((int(r["layer"]), int(r["head"])))
        if base - ablate_pct(runner, tokens, gold, chosen) >= const.BREAK_DROP:
            return chosen
    return chosen or [const.COPY_HEAD]


def locked_ops(named: list[tuple[int, int]], other: list[tuple[int, int]]) -> list[dict]:
    """Predefined before measuring. The named set should break/restore; a leftover head should not."""
    ops = [
        {"op": "ablate", "heads": list(named), "expect": "break"},
        {"op": "patch", "heads": list(named), "expect": "restore"},
    ]
    leftover = other[:1] if other else list(named[:1])
    ops.append({"op": "ablate", "heads": leftover, "expect": "hold"})
    ops.append({"op": "patch", "heads": leftover, "expect": "no_restore"})
    return ops


def run_ops(runner, tokens, gold, ops, rng) -> tuple[float, list[dict]]:
    base = accuracy(runner, tokens, gold)
    corrupt = make_corrupt(tokens, rng)
    clean_acts = runner.collect(tokens)
    hits = 0
    details = []
    for spec in ops:
        heads = [tuple(h) for h in spec["heads"]]
        expect = spec["expect"]
        if spec["op"] == "ablate":
            pct = ablate_pct(runner, tokens, gold, heads)
            ok = (base - pct) >= const.BREAK_DROP if expect == "break" else (base - pct) <= const.HOLD_DROP
        else:
            patch = {h: clean_acts[h] for h in heads if h in clean_acts}
            pct = accuracy(runner, corrupt, gold, patch=patch)
            # restore: patched corrupt matches clean gold; no_restore: it does not
            if expect == "restore":
                ok = pct >= 85.0
            else:
                ok = pct < 50.0
        hits += int(ok)
        details.append({**spec, "pct": round(pct, 4), "ok": bool(ok)})
    success = 100.0 * hits / max(len(ops), 1)
    return success, details


def random_ops(n_layers: int, n_heads: int, n: int, rng: np.random.Generator, avoid=()) -> list[dict]:
    pool = [h for h in all_heads(n_layers, n_heads) if h not in set(avoid)]
    if not pool:
        pool = all_heads(n_layers, n_heads)
    ops = []
    for _ in range(n):
        h = pool[int(rng.integers(0, len(pool)))]
        if rng.random() < 0.5:
            ops.append({"op": "ablate", "heads": [h], "expect": "break"})
        else:
            ops.append({"op": "patch", "heads": [h], "expect": "restore"})
    return ops


def evaluate_runner(runner, n: int = const.MAX_EVAL, seed: int = const.SEED, named=None) -> dict:
    rng = np.random.default_rng(seed)
    tokens, gold = make_batch(n, rng)
    task = accuracy(runner, tokens, gold)
    rows = sweep_ablation(runner, tokens, gold)
    if named is None:
        if type(runner).__name__ == "Circuit":
            named = [const.COPY_HEAD]
        else:
            named = breaking_set(runner, tokens, gold, rows)
    others = [h for h in all_heads(runner.n_layers, runner.n_heads) if h not in set(named)]
    ops = locked_ops(named, others)
    success, details = run_ops(runner, tokens, gold, ops, rng)
    joint = []
    chosen: list[tuple[int, int]] = []
    for r in rows:
        chosen.append((int(r["layer"]), int(r["head"])))
        pct = ablate_pct(runner, tokens, gold, chosen)
        joint.append(
            {
                "n": len(chosen),
                "heads": [list(h) for h in chosen],
                "ablate_pct": round(pct, 4),
                "drop": round(task - pct, 4),
            }
        )
    rand_ops = random_ops(runner.n_layers, runner.n_heads, len(ops), np.random.default_rng(seed + 7), avoid=named)
    random_pct, _ = run_ops(runner, tokens, gold, rand_ops, np.random.default_rng(seed + 11))
    return {
        "task_pct": round(task, 4),
        "success_pct": round(success, 4),
        "intervention_pct": round(success, 4),
        "random_intervention_pct": round(random_pct, 4),
        "chance_pct": round(chance_pct(), 4),
        "named_heads": named,
        "sweep": rows,
        "joint": joint,
        "ops": details,
        "n_eval": int(n),
        "vocab": const.VOCAB,
        "seq_len": const.SEQ_LEN,
        "n_layers": int(runner.n_layers),
        "n_heads": int(runner.n_heads),
        "illustrative": False,
    }
