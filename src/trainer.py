from data_loader import ShipDatasetLoader
from preprocessor import ShipPreprocessor
from model import SegmentationModel
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import load_model
from src.metrics import iou_metric, f2_metric
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

import sys
import os
sys.path.insert(0, '.')

from config import (TRAIN_IMAGES_DIR, TRAIN_METADATA_CSV, IMG_SIZE, BATCH_SIZE, RANDOM_SEED, EPOCHS, LEARNING_RATE, VALIDATION_SPLIT)

def iou_metric(y_true, y_pred, smooth=1e-6):
    # Ensure both are float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.0, tf.float32)

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

def f2_metric(y_true, y_pred, beta=2, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.0, tf.float32)

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    tp = K.sum(y_true_f * y_pred_f)
    fp = K.sum(y_pred_f) - tp
    fn = K.sum(y_true_f) - tp

    f2 = (1 + beta**2) * tp / ((1 + beta**2) * tp + beta**2 * fn + fp + smooth)
    return f2


def main():
    loader = ShipDatasetLoader(
        train_image_dir=TRAIN_IMAGES_DIR,
        masks_csv_path=TRAIN_METADATA_CSV,
        img_size=IMG_SIZE
    )

    preprocessor = ShipPreprocessor()
    train_ds, val_ds = loader.train_val_split(
        val_split=VALIDATION_SPLIT,
        batch_size=BATCH_SIZE
    )
    
    preprocessed_train_ds = preprocessor.preprocess_pipeline(train_ds, training=True, augment=True)
    preprocessed_train_ds = preprocessed_train_ds.repeat()

    model_v1 = SegmentationModel()
    model_v1.model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                           optimizer=Adam(learning_rate=LEARNING_RATE),
                           metrics=[iou_metric, f2_metric])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    checkpointer=ModelCheckpoint(filepath='model.keras', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
    tb = TensorBoard(log_dir='logs', histogram_freq=1, write_graph=1)

    steps_per_epoch = max(100, 154044 // BATCH_SIZE)
    network_history = model_v1.model.fit(
        preprocessed_train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch= steps_per_epoch,
        validation_steps=38512 // BATCH_SIZE,
        verbose=1,
        callbacks=[early_stopping, checkpointer, reduce_lr, tb]
    )

if __name__ == "__main__":
    main()