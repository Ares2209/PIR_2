"""Hyperparameters for the GCN model. Edit values here to tune training."""

# Model architecture
HIDDEN_CHANNELS = 64
DROPOUT = 0.5

# Optimization
LR = 8E-5
WEIGHT_DECAY = 5E-5
EPOCHS = 300
BATCH_SIZE = 4

# Dataset split
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SPLIT_SEED = 0

# Deterministic init / training (weights, dropout, shuffling)
SEED = 42

# Evaluation cadence (val/test run every N epochs; always on last epoch)
EVAL_EVERY = 5

# Evaluation cadence (val/test run every N epochs; also on epoch 1 and last)
EVAL_EVERY = 5

# Early stopping
PATIENCE = 50
MIN_DELTA = 1e-4

# Focal loss (7 classes: violet, blue, yellow, orange, red, dark_red, occluded)
# Weights from inverse-sqrt frequency on the global distribution, min-normalized.
# dark_red (0.028%) gets the largest weight; occluded (majority) the smallest.
FOCAL_ALPHA = [1.84, 1.46, 2.95, 5.79, 14.34, 25.0, 1.0]
FOCAL_GAMMA = 2.0
