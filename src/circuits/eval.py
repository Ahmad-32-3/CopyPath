from __future__ import annotations

import json
from pathlib import Path

from . import const
from .intervene import evaluate_runner
from .train import circuit_runner, train_tiny, untrained_runner


def evaluate_circuit(n: int = const.MAX_EVAL, seed: int = const.SEED) -> dict:
    return evaluate_runner(circuit_runner(), n=n, seed=seed, named=[const.COPY_HEAD])


def evaluate_untrained(n: int = 64, seed: int = const.SEED) -> dict:
    m = evaluate_runner(untrained_runner(), n=n, seed=seed, named=[const.COPY_HEAD])
    return m


def evaluate_trained(steps: int = const.MAX_TRAIN_STEPS) -> dict:
    runner, note = train_tiny(steps=steps)
    if runner is None:
        m = evaluate_circuit()
        m["illustrative"] = True
        m["blocker"] = note
        m["source"] = "circuit"
        return m
    m = evaluate_runner(runner)
    m["source"] = "trained"
    m["train_note"] = note
    return m


def print_report(m: dict):
    tag = "ILLUSTRATIVE" if m.get("illustrative") else "measured"
    print(
        f"task_pct {m['task_pct']:.1f}  |  intervention_pct {m['intervention_pct']:.1f}  "
        f"|  success_pct {m['success_pct']:.1f}  |  random_intervention_pct "
        f"{m['random_intervention_pct']:.1f}  |  chance_pct {m['chance_pct']:.1f}  [{tag}]"
    )
    print(
        f"named {m.get('named_heads')}  n_eval {m['n_eval']}  "
        f"layers {m['n_layers']} heads {m['n_heads']}  source {m.get('source', 'circuit')}"
    )


def write_metrics(m: dict, path: str = const.METRICS_PATH):
    out = {k: v for k, v in m.items() if k not in {"ops"}}
    Path(path).write_text(json.dumps(out, indent=2), encoding="utf-8")
