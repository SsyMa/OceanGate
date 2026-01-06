import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Conv2DTranspose, Concatenate
import sys
import os
sys.path.insert(0, '.')

from config import (IMG_HEIGHT, IMG_WIDTH)

class SegmentationModel(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Encoder
        self.conv1 = Conv2D(filters=16, kernel_size=3, activation='relu', padding ='same')
        self.bNorm1 = BatchNormalization()
        self.downScale1 = Conv2D(filters=32, kernel_size=3, activation='relu', padding ='same', strides = 2) # Strided conv for downscaling
        self.bNorm2 = BatchNormalization()
        self.conv2 = Conv2D(filters=32, kernel_size=3, activation='relu', padding ='same')
        self.bNorm3 = BatchNormalization()
        self.downScale2 = Conv2D(filters=64, kernel_size=3, activation='relu', padding ='same', strides = 2)
        self.bNorm4 = BatchNormalization()
        self.conv3 = Conv2D(filters=64, kernel_size=3, activation='relu', padding ='same')
        self.bNorm5 = BatchNormalization()
        self.downScale3 = Conv2D(filters=128, kernel_size=3, activation='relu', padding ='same', strides = 2)
        self.bNorm6 = BatchNormalization()
        self.conv4 = Conv2D(filters=128, kernel_size=3, activation='relu', padding ='same')
        self.bNorm7 = BatchNormalization()
        self.downScale4 = Conv2D(filters=256, kernel_size=3, activation='relu', padding ='same', strides = 2)
        self.bNorm8 = BatchNormalization()
        self.conv5 = Conv2D(filters=256, kernel_size=3, activation='relu', padding ='same')
        self.bNorm9 = BatchNormalization()

        # Decoder
        self.trConv1 = Conv2DTranspose(filters=128,  kernel_size=3,  strides=2, padding='same', activation='relu')
        self.decConv1 = Conv2D(128, 3, activation='relu', padding='same') # Refining conv after skip connection
        self.bNorm10 = BatchNormalization()

        self.trConv2 = Conv2DTranspose(filters=64,  kernel_size=3,  strides=2, padding='same', activation='relu')
        self.decConv2 = Conv2D(64, 3, activation='relu', padding='same')
        self.bNorm11 = BatchNormalization()

        self.trConv3 = Conv2DTranspose(filters=32,  kernel_size=3,  strides=2, padding='same', activation='relu')
        self.decConv3 = Conv2D(32, 3, activation='relu', padding='same')
        self.bNorm12 = BatchNormalization()

        self.trConv4 = Conv2DTranspose(filters=16,  kernel_size=3,  strides=2, padding='same', activation='relu')
        self.decConv4 = Conv2D(16, 3, activation='relu', padding='same')
        self.bNorm13 = BatchNormalization()

        # Classifier
        self.classifier = Conv2D(filters=1, kernel_size=1, padding = 'same')

    def call(self, inputs):
        # Define forward pass with skip connections (U-net like architecture)

        # Decoder
        x = self.conv1(inputs)
        x = self.bNorm1(x)
        skip1 = x

        x = self.downScale1(x)
        x = self.bNorm2(x)
        x = self.conv2(x)
        x = self.bNorm3(x)
        skip2 = x

        x = self.downScale2(x)
        x = self.bNorm4(x)
        x = self.conv3(x)
        x = self.bNorm5(x)
        skip3 = x

        x = self.downScale3(x)
        x = self.bNorm6(x)
        x = self.conv4(x)
        x = self.bNorm7(x)
        skip4 = x

        x = self.downScale4(x)
        x = self.bNorm8(x)
        x = self.conv5(x)
        x = self.bNorm9(x)

        # Decoder with added skip connections
        x = self.trConv1(x)
        x = Concatenate(axis=-1)([x, skip4])
        x = self.decConv1(x)
        x = self.bNorm10(x)

        x = self.trConv2(x)
        x = Concatenate(axis=-1)([x, skip3])
        x = self.decConv2(x)
        x = self.bNorm11(x)

        x = self.trConv3(x)
        x = Concatenate(axis=-1)([x, skip2])
        x = self.decConv3(x)
        x = self.bNorm12(x)

        x = self.trConv4(x)
        x = Concatenate(axis=-1)([x, skip1])
        x = self.decConv4(x)
        x = self.bNorm13(x)

        return self.classifier(x)


if __name__ == "__main__":
    pass
