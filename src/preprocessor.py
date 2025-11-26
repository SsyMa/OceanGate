"""
Image preprocessing and augmentation for Airbus Ship Detection.

This module handles:
- Image augmentation (training only)
- Test-time augmentation (TTA)
"""
import sys
import os
sys.path.insert(0, '.')

import tensorflow as tf
from typing import Tuple
from sklearn.model_selection import train_test_split

from config import (TRAIN_IMAGES_DIR, TRAIN_METADATA_CSV, IMG_SIZE, BATCH_SIZE, RANDOM_SEED)

class ShipPreprocessor:
    """Preprocessing pipeline for Airbus Ship Detection."""
    
    def __init__(self, img_size: Tuple[int, int] = IMG_SIZE):
        self.img_size = img_size
    
    def training_augmentation(self, image: tf.Tensor, mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Strong augmentation pipeline for training."""
        
        if tf.random.uniform(()) > 0.5:
            # Horizontal flip
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)
        
        if tf.random.uniform(()) > 0.5:
            # Vertical flip
            image = tf.image.flip_up_down(image)
            mask = tf.image.flip_up_down(mask)
        
        # Random rotation (0, 90, 180, 270 degrees)
        rotation = tf.random.uniform([], 0, 4, dtype=tf.int32)
        image = tf.image.rot90(image, k=rotation)
        mask = tf.image.rot90(mask, k=rotation)
        
        # Random brightness/contrast
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        
        # Random hue/saturation
        image = tf.image.random_hue(image, 0.1)
        image = tf.image.random_saturation(image, 0.8, 1.2)
        
        # Clamp to valid range
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image, mask
    
    def test_augmentation(self, image: tf.Tensor) -> list:
        """
        Test-time augmentation (TTA): 8 augmented versions.

        TTA boosts test accuracy by also performing inference in augmented versions 
        of the original Image and then avarage out the result.

        This can help inference performance since the model essentially avaraging out its own errors.

        These augmentations are deterministic and reversible if needed.
        """
        aug_images = []
        
        # Original
        aug_images.append(image)
        
        # Horizontal flip
        aug_images.append(tf.image.flip_left_right(image))
        
        # Vertical flip
        aug_images.append(tf.image.flip_up_down(image))
        
        # Both flips
        aug_images.append(tf.image.flip_left_right(tf.image.flip_up_down(image)))
        
        # 90 degree rotations
        aug_images.append(tf.image.rot90(image))
        aug_images.append(tf.image.rot90(tf.image.flip_left_right(image)))
        
        # 270 degree rotations  
        aug_images.append(tf.image.rot90(image, k=3))
        aug_images.append(tf.image.rot90(tf.image.flip_left_right(image), k=3))
        
        return aug_images
    
    def preprocess_pipeline(
        self, 
        dataset: tf.data.Dataset, 
        training: bool = True,
        augment: bool = True
    ) -> tf.data.Dataset:
        """Complete preprocessing pipeline."""
        def preprocess_fn(image, mask):
            
            if training and augment:
                image, mask = self.training_augmentation(image, mask)
            
            return image, mask
        
        return dataset.map(
            preprocess_fn,
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)


def main():
    """Test the preprocessor."""
    from src.data_loader import ShipDatasetLoader
    
    # Test with data loader
    loader = ShipDatasetLoader(
        train_image_dir=TRAIN_IMAGES_DIR,
        masks_csv_path=TRAIN_METADATA_CSV,
        img_size=IMG_SIZE
    )
    
    preprocessor = ShipPreprocessor()
    
    # Test augmentation pipeline
    train_ds, val_ds = loader.train_val_split(batch_size=BATCH_SIZE)
    preprocessed_train_ds = preprocessor.preprocess_pipeline(train_ds, training=True, augment=True)

    # To train keras model, use:
    # model.fit(preprocessed_train_ds, validation_data=val_ds, epochs=50)
    
    print("Testing preprocessing pipeline:")
    for images, masks in preprocessed_train_ds.take(1):
        print(f"  Images shape: {images.shape}, range: [{tf.reduce_min(images):.3f}, {tf.reduce_max(images):.3f}]")
        print(f"  Masks shape: {masks.shape}")
    
    print("Preprocessor test successful!")


if __name__ == "__main__":
    main()