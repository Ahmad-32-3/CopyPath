# Frozen protocol. Change only with one ADR line in DESIGN.md, then re-measure.

VOCAB = 16
SEQ_LEN = 4  # [x, y, z, x] -> y (induction-style copy)
N_LAYERS = 1
N_HEADS = 4
D_MODEL = 32
D_FF = 0  # attention-only: copy cannot hide in an MLP
MAX_EVAL = 256
MAX_TRAIN_STEPS = 400
BATCH = 64
SEED = 0
LR = 3e-3

# Compiled circuit (tests + fallback): copy lives in layer 0, head 0.
COPY_HEAD = (0, 0)
OTHER_HEAD = (0, 1)

BREAK_DROP = 20.0  # task_pct drop that counts as a predicted break
HOLD_DROP = 5.0  # drop below this counts as a predicted hold

METRICS_PATH = "metrics.json"
