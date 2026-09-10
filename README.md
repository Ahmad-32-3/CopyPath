# CopyPath

A tiny network can get every answer right on a simple copy puzzle and still leave me unsure where it stores that trick.

I train a small transformer to copy a token, then I turn off or swap the pieces I claimed were doing the copy. If my story is real, those edits should change the answer in a predicted way. Random pieces should not. The number I trust is how often those checks behave as claimed.

Headline is intervention recovery rate, printed next to task accuracy and a random-intervention baseline. Task fit should be near ceiling on the toy. I am checking whether I can point at the algorithm inside a small net.

## Data

Synthetic algorithmic task (modular addition or induction-style copy). Tiny transformer.

## Run

```bash
python -m pytest tests/ -q
python scripts/run.py
npm --prefix web install
npm --prefix web run dev
```

`scripts/run.py` prints task accuracy and intervention recovery. A random patch that still "recovers" the algorithm must fail in pytest.

## Layout

- `src/` task, model, interventions
- `scripts/run.py`
- `tests/` random-patch and untrained-task checks
- `web/` which heads matter, plus task vs intervention vs random
