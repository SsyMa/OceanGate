import tensorflow as tf
from tensorflow.keras import layers, models
import config

def build_model(input_shape=(*config.IMG_SIZE, 3), num_classes=config.NUM_CLASSES, freeze_encoder=False):
    """
    ConvNeXt Tiny alapú, skip-mentes U-Net-szerű modell szegmentációhoz.
    """

    # Encoder: ConvNeXt Tiny, pretrained
    base_model = tf.keras.applications.ConvNeXtTiny(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling=None
    )
    
    if freeze_encoder:
        base_model.trainable = False

    # Output az encoderből
    x = base_model.output  # shape: (None, H/32, W/32, 768)

    # Decoder (Upsampling blokkok)
    # 3x3 conv + upsampling
    x = layers.Conv2D(512, (3,3), padding='same', activation='relu')(x)
    x = layers.UpSampling2D(size=(2,2))(x)  # H/16 x W/16

    x = layers.Conv2D(256, (3,3), padding='same', activation='relu')(x)
    x = layers.UpSampling2D(size=(2,2))(x)  # H/8 x W/8

    x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = layers.UpSampling2D(size=(2,2))(x)  # H/4 x W/4

    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = layers.UpSampling2D(size=(2,2))(x)  # H/2 x W/2

    x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(x)
    x = layers.UpSampling2D(size=(2,2))(x)  # H x W

    # Output layer
    outputs = layers.Conv2D(num_classes, (1,1), activation='sigmoid')(x)

    model = models.Model(inputs=base_model.input, outputs=outputs)
    return model

def unfreeze_encoder(model):
    model.trainable = True

if __name__ == "__main__":
    model = build_model()
    model.summary()
