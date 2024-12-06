# config.py

# Data parameters
DATASET_NAME = "Housing_consumption"  # Options: "Mackey_glass", "Lorenz96", "Housing_consumption", "Dream3"
DATA_SIZE = 50               # Number of time steps in the dataset

# Mackey_glass parameters
TAU = 17
DELTA_T = 0.1
NUM_ATTRIBUTE = 4
BETA = 0.2
GAMMA = 0.01
MACKEY_N = 10
SEQ_LEN = 10

# Lorenz96 parameters

LORENZ96_N = 4
LORENZ96_F = 10
LORENZ96_T_END = 10
LORENZ96_T_POINTS = 100

# Training parameters
BATCH_SIZE = 32               # Batch size for training
NUM_EPOCHS = 20               # Number of epochs
LEARNING_RATE = 0.001         # Learning rate for optimizer
DROPOUT_RATE = 0.1            # Dropout rate in Transformer
LAMBDA_REG = 0.1


# Transformer model parameters
D_MODEL = 128                 # Embedding dimension
NHEAD = 4                     # Number of attention heads
NUM_LAYERS = 3                # Number of Transformer layers
DIM_FEEDFORWARD = 512         # Feedforward layer dimension
DILATION = 2

# PCMCI parameters
PCMCI_METHOD = 'CMIknn'         # GDPC or CMIknn
TAU_MAX = 3                   # Maximum time lag for PCMCI
ALPHA_LEVEL = 0.1             # Significance level for PCMCI
SIGNIFICANCE_METHOD = "shuffle_test"  # Options: "analytic", "shuffle_test", "bootstrap"
PC_ALPHA = 0.05              # Significance level in PC
PC_VERBOSITY = 1             # the detail

# Transformer model parameters
TRANSFORMER_CONFIG = {
    "DREAM3": {
        "input_dim": 3,  # 3 features: wt, gcn4, leu3
        "output_dim": 1,  # 8 time points for gat1
        "num_layers": 3,
        "d_model": 128,
        "nhead": 4,
        "dim_feedforward": 256,
        "dilation": 2,
        "dropout": 0.1,
    },
    "Household": {
        "input_dim": 6,  # Example for another dataset
        "output_dim": 1,
        "num_layers": 4,
        "d_model": 64,
        "nhead": 8,
        "dim_feedforward": 256,
        "dilation": 1,
        "dropout": 0.2,
    },

    "Mackey_glass": {
        "input_dim": 3,  # Example for another dataset
        "output_dim": 1,
        "num_layers": 4,
        "d_model": 64,
        "nhead": 8,
        "dim_feedforward": 256,
        "dilation": 1,
        "dropout": 0.2,
    },

    "Lorenz96": {
        "input_dim": 3,  # Example for another dataset
        "output_dim": 1,
        "num_layers": 4,
        "d_model": 64,
        "nhead": 8,
        "dim_feedforward": 256,
        "dilation": 1,
        "dropout": 0.2,
    },
}