"""Hyperparameters for the EdgeSAGE model. Edit values here to tune training."""

# Model architecture
HIDDEN_CHANNELS = 64
NUM_LAYERS = 5
DROPOUT = 0.2

# Optimization
LR = 1E-4
WEIGHT_DECAY = 5E-5
EPOCHS = 300
BATCH_SIZE = 4
GRAD_CLIP = 1.0

# Dataset split (group-split by map name to prevent inter-city leakage)
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SPLIT_SEED = 0

# Deterministic init / training (weights, dropout, shuffling)
SEED = 42

# Evaluation cadence (val/test run every N epochs; also on epoch 1 and last)
EVAL_EVERY = 5

# Early stopping
PATIENCE = 50
MIN_DELTA = 1e-4

# LR scheduler (ReduceLROnPlateau on val MCC)
LR_FACTOR = 0.5
LR_PATIENCE = 10
LR_MIN = 1e-6

# Focal loss — alpha is auto-computed at train time from the training-split
# class distribution (inverse-sqrt freq, min-normalized). Only gamma is fixed.
FOCAL_GAMMA = 2.0

# Checkpoint rotation: how many previous best checkpoints to keep in ./old/
KEEP_OLD_CHECKPOINTS = 3
