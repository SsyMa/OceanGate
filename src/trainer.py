from data_loader import ShipDatasetLoader
from preprocessor import ShipPreprocessor
from model import SegmentationModel
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from src.metrics import iou_metric, f2_metric

import sys
import os
sys.path.insert(0, '.')

from config import (TRAIN_IMAGES_DIR, TRAIN_METADATA_CSV, IMG_SIZE, BATCH_SIZE, RANDOM_SEED, EPOCHS, LEARNING_RATE, VALIDATION_SPLIT)

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

    tb = TensorBoard(log_dir='logs', histogram_freq=1, write_graph=1)
    checkpointer=ModelCheckpoint(filepath='model.keras', save_best_only=True, verbose=1)

    steps_per_epoch = max(100, 154044 // BATCH_SIZE)

    network_history = model_v1.model.fit(
        preprocessed_train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch= steps_per_epoch,
        validation_steps=38512 // BATCH_SIZE,
        verbose=1,
        callbacks=[tb, checkpointer])


if __name__ == "__main__":
    main()