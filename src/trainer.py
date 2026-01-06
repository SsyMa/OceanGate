from data_loader import ShipDatasetLoader
from preprocessor import ShipPreprocessor
from model import SegmentationModel
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from src.metrics import iou_metric, f2_metric
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

import sys
import os
sys.path.insert(0, '.')

from config import (TRAIN_IMAGES_DIR, TRAIN_METADATA_CSV, IMG_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, VALIDATION_SPLIT)

'''
Main script to train network
'''

def main():

    # Load dataset and create train/validation split
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
    
    # Create prepocessed dataset with augmented images
    preprocessed_train_ds = preprocessor.preprocess_pipeline(train_ds, training=True, augment=True)

    # Create and compile model with the custom metrics
    # - from_logits needed because the last layer of the network doesn't have activation.
    model_v1 = SegmentationModel()
    model_v1.model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                           optimizer=Adam(learning_rate=LEARNING_RATE),
                           metrics=[iou_metric, f2_metric])

    # Callbacks for training
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    checkpointer=ModelCheckpoint(filepath='model.keras', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)
    tb = TensorBoard(log_dir='logs', histogram_freq=1, write_graph=1)

    network_history = model_v1.model.fit(
        preprocessed_train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1,
        callbacks=[early_stopping, checkpointer, reduce_lr, tb]
    )

if __name__ == "__main__":
    main()