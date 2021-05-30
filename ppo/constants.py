# ENV
STATE_DIM = 4
ACT_DIM = 2

# TRAINING
CRITIC_COEFF = 1 # How heavily to weight value networks loss
EPOCHS_PER_ROLLOUT = 3
BATCH_SIZE = -1
MAX_GRAD_NORM = 5
ENT_COEFF = 0.01

# HYPERPARAMS
EPSILON = 0.2 # Clip radius for PPO loss
GAMMA = 0.99 # Discount factor
LEARNING_RATE = 2e-3
TAU = 0.95 # For GAE in util.py