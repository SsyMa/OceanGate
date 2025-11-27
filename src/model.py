import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Conv2DTranspose
from tensorflow.keras.models import Sequential
import sys
import os
sys.path.insert(0, '.')

from config import (TRAIN_IMAGES_DIR, TRAIN_METADATA_CSV, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, RANDOM_SEED)

class SegmentationModel():
    def __init__(self):
        self.model = Sequential()
        # Encoder
        self.model.add(Conv2D(filters=16, kernel_size=3, activation='relu', padding ='same', input_shape=(IMG_HEIGHT,IMG_WIDTH,3)))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=32, kernel_size=3, activation='relu', padding ='same', strides = 2))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=32, kernel_size=3, activation='relu', padding ='same'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding ='same', strides = 2))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding ='same'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding ='same', strides = 2))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding ='same'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=256, kernel_size=3, activation='relu', padding ='same', strides = 2))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters=256, kernel_size=3, activation='relu', padding ='same'))
        self.model.add(BatchNormalization())

        # Decoder
        self.model.add(Conv2DTranspose(filters=128,  kernel_size=3,  strides=2, padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2DTranspose(filters=64,  kernel_size=3,  strides=2, padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2DTranspose(filters=32,  kernel_size=3,  strides=2, padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2DTranspose(filters=16,  kernel_size=3,  strides=2, padding='same', activation='relu'))
        self.model.add(BatchNormalization())

        # Classifier
        self.model.add(Conv2D(filters=1, kernel_size=5, padding = 'same'))


if __name__ == "__main__":
    pass
