# train.py
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import backend as K
import numpy as np
import config
from data_loading import DataLoader
from preprocessing import ShipDataGenerator, create_tf_dataset
from model import build_model
import os

# -----------------------------
# Mixed precision for speed
# -----------------------------
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# -----------------------------
# Metrics for segmentation
# -----------------------------
def iou_metric(y_true, y_pred, smooth=1e-6):
    # Ensure both are float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

def f2_metric(y_true, y_pred, beta=2, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    tp = K.sum(y_true_f * y_pred_f)
    fp = K.sum(y_pred_f) - tp
    fn = K.sum(y_true_f) - tp
    
    f2 = (1 + beta**2) * tp / ((1 + beta**2) * tp + beta**2 * fn + fp + smooth)
    return f2

# -----------------------------
# Load and prepare data
# -----------------------------
import numpy as np
from data_loading import DataLoader
from preprocessing import ShipDataGenerator
import config

# 1. Initialize the data loader and load metadata
loader = DataLoader()
df = loader.load_metadata()  # metadata contains image IDs and masks

# 2. Get unique image IDs
all_ids = df['ImageId'].unique()

# 3. Split into train, validation, and test sets (example split, can use sklearn)
from sklearn.model_selection import train_test_split

train_ids, temp_ids = train_test_split(
    all_ids,
    test_size=config.VALIDATION_SPLIT + config.TEST_SPLIT,
    random_state=config.RANDOM_SEED
)

val_ids, test_ids = train_test_split(
    temp_ids,
    test_size=config.TEST_SPLIT / (config.VALIDATION_SPLIT + config.TEST_SPLIT),
    random_state=config.RANDOM_SEED
)

print(f"Train: {len(train_ids)}, Validation: {len(val_ids)}, Test: {len(test_ids)}")

# 4. Create data generators
# Prefer tf.data pipeline (faster). create_tf_dataset uses the existing DataLoader cache
train_ds = create_tf_dataset(train_ids, data_loader=loader, batch_size=config.BATCH_SIZE,
                             img_size=config.IMG_SIZE, shuffle=True, augment=True)

val_ds = create_tf_dataset(val_ids, data_loader=loader, batch_size=config.BATCH_SIZE,
                           img_size=config.IMG_SIZE, shuffle=False, augment=False)

test_ds = create_tf_dataset(test_ids, data_loader=loader, batch_size=config.BATCH_SIZE,
                            img_size=config.IMG_SIZE, shuffle=False, augment=False)

# -----------------------------
# Build model
# -----------------------------
model = build_model(freeze_encoder=True)
print(model.summary())

# -----------------------------
# Callbacks
# -----------------------------
checkpoint_cb = ModelCheckpoint(
    filepath=config.MODELS_DIR + '/best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

reduce_lr_cb = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1,
    min_lr=1e-6
)

early_stop_cb = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1,
    restore_best_weights=True
)

# -----------------------------
# Phase 1: Train decoder only
# -----------------------------
print("Phase 1: Training decoder only (freeze base model)")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=[iou_metric, f2_metric]
)

history_phase1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=config.EPOCHS_PHASE1,
    callbacks=[checkpoint_cb, reduce_lr_cb, early_stop_cb]
)

# -----------------------------
# Phase 2: Fine-tuning full model
# -----------------------------
print("Phase 2: Fine-tuning full model")
model.trainable = True  # Unfreeze entire model

# Lower learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=[iou_metric, f2_metric]
)

history_phase2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=config.EPOCHS_PHASE2,
    callbacks=[checkpoint_cb, reduce_lr_cb, early_stop_cb]
)

print("Training complete. Best model saved to:", config.MODEL_SAVE_PATH)
