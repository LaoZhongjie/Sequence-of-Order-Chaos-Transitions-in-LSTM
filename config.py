"""
Configuration file for LSTM Multiple Descents Experiment
Contains all hyperparameters and settings used in the paper
"""

# Model hyperparameters (from Table I in paper)
EMBEDDING_DIM = 32
LSTM_HIDDEN_SIZE = 60
NUM_CLASSES = 1  # Binary sentiment classification

# Training parameters
LEARNING_RATE = 0.0005
BATCH_SIZE = 32
MAX_EPOCHS = 1000  # Paper uses 10000, but we start with 1000 for feasibility
SEQUENCE_LENGTH = 500  # Fixed length for padding/truncating reviews

# Data split
TRAIN_RATIO = 0.7
TEST_RATIO = 0.3

# Asymptotic analysis parameters
ASYMPTOTIC_TIMESTEPS = 1600  # T = 1599 in paper (1600 total)
REVIEW_TIMESTEPS = 500  # First 500 timesteps use real review data
ZERO_INPUT_TIMESTEPS = 1100  # Remaining timesteps use zero input (1600-500)
NUM_TEST_SAMPLES = 500  # Number of test samples for asymptotic analysis
MACHINE_PRECISION_THRESHOLD = -15  # exp(-15) threshold for numerical precision
START_EPOCH = 1
END_EPOCH = 1000

# Perturbation parameters
NOISE_SCALE = 1e-3  # Scale for Gaussian noise perturbation

# Device and reproducibility
RANDOM_SEED = 42
DEVICE = 'cuda'  # Change to 'cpu' if no GPU available

# File paths
DATA_PATH = './data/'
RESULTS_PATH = './results/'
CHECKPOINT_PATH = './checkpoints/'

# Visualization parameters
SAVE_FIGURES = True
FIGURE_DPI = 300