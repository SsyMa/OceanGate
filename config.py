import os

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Data paths
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, 'train_v2')
TEST_IMAGES_DIR = os.path.join(DATA_DIR, 'test_v2')
TRAIN_METADATA_CSV = os.path.join(DATA_DIR, 'train_ship_segmentations_v2.csv')

# ============================================================================
# IMAGE PROCESSING PARAMETERS
# ============================================================================
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.15

# ============================================================================
# SEED FOR REPRODUCIBILITY
# ============================================================================
RANDOM_SEED = 42