import numpy as np
import pytest

from circuits import const
from circuits.eval import evaluate_circuit, evaluate_untrained
from circuits.intervene import ablate_pct, accuracy, recover_pct
from circuits.task import make_batch, make_corrupt
from circuits.train import circuit_runner, untrained_runner


def test_untrained_task_accuracy_fails():
    rng = np.random.default_rng(0)
    tokens, gold = make_batch(64, rng)
    pct = accuracy(untrained_runner(), tokens, gold)
    assert pct < 40, f"untrained task_pct={pct}"


def test_bogus_patch_cannot_pass():
    rng = np.random.default_rng(1)
    tokens, gold = make_batch(64, rng)
    circuit = circuit_runner()
    bogus = recover_pct(circuit, tokens, gold, [const.OTHER_HEAD], rng)
    named = recover_pct(circuit, tokens, gold, [const.COPY_HEAD], np.random.default_rng(2))
    assert bogus < 50, f"bogus patch recovered {bogus}"
    assert named >= 85, f"named patch only {named}"
    assert named > bogus


def test_ablate_copy_breaks_ablate_other_holds():
    rng = np.random.default_rng(3)
    tokens, gold = make_batch(64, rng)
    circuit = circuit_runner()
    base = accuracy(circuit, tokens, gold)
    broken = ablate_pct(circuit, tokens, gold, [const.COPY_HEAD])
    held = ablate_pct(circuit, tokens, gold, [const.OTHER_HEAD])
    assert base >= 99
    assert (base - broken) >= const.BREAK_DROP
    assert (base - held) <= const.HOLD_DROP


def test_corrupt_changes_the_copied_token():
    rng = np.random.default_rng(4)
    tokens, gold = make_batch(32, rng)
    corrupt = make_corrupt(tokens, rng)
    assert not np.array_equal(corrupt[:, 1], tokens[:, 1])
    assert np.array_equal(corrupt[:, 0], tokens[:, 0])
    assert np.array_equal(corrupt[:, -1], tokens[:, -1])
    assert not np.array_equal(corrupt[:, 1], gold)


def test_evaluate_beats_random_on_locked_suite():
    m = evaluate_circuit(n=64, seed=5)
    assert m["task_pct"] >= 99
    assert m["success_pct"] >= 85
    assert m["success_pct"] > m["random_intervention_pct"]
    assert m["n_eval"] <= const.MAX_EVAL
    assert m["seq_len"] == const.SEQ_LEN
    assert m["vocab"] == const.VOCAB


def test_untrained_evaluate_is_not_the_headline():
    m = evaluate_untrained(n=32, seed=6)
    assert m["task_pct"] < 40
