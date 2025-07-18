# ISPRS Potsdam dataset specific configurations
ISPRS_COLORMAP = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
                  [255, 255, 0], [255, 0, 0]]

ISPRS_CLASSES = ['Impervious Surfaces', 'Building', 'Low Vegetation', 'Tree',
                 'Car', 'Clutter/Background']

NUM_CLASSES = len(ISPRS_CLASSES)
IGNORE_INDEX = 255

HPARAM_COMBINATIONS = list(itertools.product(
    [0.001, 0.0001],  # Learning rates
    [8, 16],          # Batch sizes
    ['Adam', 'SGD']   # Optimizers
))

# Training parameters
N_EPOCHS = 10
PATIENCE = 2 # Early stopping patience

# Data split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Other global settings
SEED = 77 # For reproducibility
