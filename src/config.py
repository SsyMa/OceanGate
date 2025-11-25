import os

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = os.path.dirname('.')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Data paths
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, 'train_v2')
TEST_IMAGES_DIR = os.path.join(DATA_DIR, 'test_v2')
TRAIN_METADATA_CSV = os.path.join(DATA_DIR, 'train_ship_segmentations_v2.csv')

# Output directories
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')

# ============================================================================
# IMAGE PROCESSING PARAMETERS
# ============================================================================
ORIGINAL_IMG_HEIGHT = 768
ORIGINAL_IMG_WIDTH = 768
IMG_HEIGHT = 384
IMG_WIDTH = 384
IMG_CHANNELS = 3
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
BACKBONE = 'convnext_tiny'
USE_PRETRAINED = True
NUM_CLASSES = 1  # Binary segmentation (ship vs no-ship)

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================
BATCH_SIZE = 16
EPOCHS = 50
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 10
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.1

# Early stopping
PATIENCE = 10
MIN_DELTA = 0.001

# ============================================================================
# DATA AUGMENTATION
# ============================================================================
ROTATION_RANGE = 20.0
WIDTH_SHIFT_RANGE = 0.1
HEIGHT_SHIFT_RANGE = 0.1
ZOOM_RANGE = 0.2
HORIZONTAL_FLIP = True
VERTICAL_FLIP = True
BRIGHTNESS_RANGE = (0.8, 1.2)

# ============================================================================
# SEED FOR REPRODUCIBILITY
# ============================================================================
RANDOM_SEED = 42

# Create necessary directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)