"""
Data loading utilities for Airbus Ship Detection challenge.

This module provides efficient data loading using TensorFlow's tf.data API
with lazy loading, prefetching, and RLE mask decoding. It implements best
practices for handling large-scale image datasets without memory overflow.

The module handles:
    - Run-Length Encoded (RLE) mask decoding
    - Stratified train/validation splitting
    - Class balancing for imbalanced datasets
    - GPU-optimized data pipeline with prefetching
    - Memory-efficient lazy loading

Example:
    Basic usage with train/validation split::

        from src.data_loader import ShipDatasetLoader

        loader = ShipDatasetLoader(
            train_image_dir="../data/train_v2",
            masks_csv_path="../data/train_ship_segmentations_v2.csv",
            img_size=(256, 256)
        )

        train_ds, val_ds = loader.train_val_split(
            val_split=0.2,
            batch_size=16
        )

        # Use with Keras model
        model.fit(train_ds, validation_data=val_ds, epochs=50)

Notes:
    - Images are loaded lazily (on-demand) to avoid memory overflow
    - RLE masks are decoded on-the-fly during data loading
    - The pipeline uses TensorFlow's AUTOTUNE for optimal performance
"""
import sys
import os
sys.path.insert(0, '.')

import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from sklearn.model_selection import train_test_split

from config import TRAIN_IMAGES_DIR, TRAIN_METADATA_CSV, IMG_SIZE, BATCH_SIZE, VALIDATION_SPLIT, RANDOM_SEED


class ShipDatasetLoader:
    """
    Airbus Ship Detection dataset loader for TensorFlow/Keras.

    This class implements best practices for large-scale image dataset loading:
        - Lazy loading using tf.data.Dataset API (no memory overflow)
        - Prefetching for GPU optimization (GPU never waits for data)
        - On-the-fly RLE mask decoding
        - Memory-efficient pipeline with parallel processing
        - Stratified train/validation splitting

    The data pipeline is optimized to:
        1. Never load entire dataset into memory (use tf.data.Dataset)
        2. Use AUTOTUNE for parallel processing and prefetching
        3. Decode masks on-the-fly, not pre-loaded
        4. Normalize images to [0, 1] range
        5. Use nearest neighbor interpolation for binary masks

    Attributes:
        train_image_dir (Path): Directory containing training images.
        img_size (Tuple[int, int]): Target image size as (height, width).
        seed (int): Random seed for reproducibility.
        df (pd.DataFrame): Original dataframe from CSV.
        df_grouped (pd.DataFrame): Grouped dataframe with one row per image.

    Example:
        loader = ShipDatasetLoader(
            train_image_dir="../data/train_v2",
            masks_csv_path="../data/train_ship_segmentations_v2.csv"
        )
        train_ds = loader.create_dataset(batch_size=8)
        for images, masks in train_ds.take(1):
            print(images.shape, masks.shape)
    """

    def __init__(
        self,
        train_image_dir: str,
        masks_csv_path: str,
        img_size: Tuple[int, int] = (256, 256),
        seed: int = 42
    ) -> None:
        """
        Initialize the dataset loader.

        Args:
            train_image_dir: Path to directory containing training images.
                Example: "../data/train_v2"
                This should point to the folder with .jpg files.
            masks_csv_path: Path to CSV file with RLE encoded masks.
                Example: "../data/train_ship_segmentations_v2.csv"
                This CSV contains ImageId and EncodedPixels columns.
            img_size: Target image size as (height, width) tuple.
                Images will be resized to this dimension.
                Default is (768, 768) which matches original image size.
            seed: Random seed for reproducibility of shuffling and splits.
                Default is 42.

        Raises:
            FileNotFoundError: If train_image_dir or masks_csv_path does not exist.
            ValueError: If masks_csv_path is not a valid CSV file.
        """
        self.train_image_dir = Path(train_image_dir)
        self.img_size = img_size
        self.seed = seed

        # Validate paths
        if not self.train_image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {train_image_dir}"
            )

        if not Path(masks_csv_path).exists():
            raise FileNotFoundError(
                f"CSV file not found: {masks_csv_path}"
            )

        # Load and preprocess the CSV file containing mask annotations
        self.df = pd.read_csv(masks_csv_path)
        self._preprocess_dataframe()

    def _preprocess_dataframe(self) -> None:
        """
        Preprocess the CSV dataframe.

        The original CSV has one row per ship instance, but one image can contain
        multiple ships. This method aggregates all RLE masks per image and creates
        a binary flag for ship presence.

        Processing steps:
            1. Mark images with/without ships (NaN in EncodedPixels means no ship)
            2. Group by ImageId to aggregate all masks for each image
            3. Create a list of RLE masks per image
            4. Store ship presence flag per image

        Side effects:
            Creates self.df_grouped attribute with aggregated data.
        """
        # NaN in EncodedPixels column means the image has no ships
        self.df['has_ship'] = self.df['EncodedPixels'].notna()

        # Group by ImageId: one image can have multiple ships (multiple RLE masks)
        # We aggregate all RLE masks into a list for each unique image
        self.df_grouped = self.df.groupby('ImageId').agg({
            'EncodedPixels': lambda x: list(x.dropna()),
            'has_ship': 'max'
        }).reset_index()

        # Print dataset statistics
        total_images = len(self.df_grouped)
        images_with_ships = self.df_grouped['has_ship'].sum()
        images_without_ships = (~self.df_grouped['has_ship']).sum()

        print(f"Dataset statistics:")
        print(f"  Total unique images: {total_images}")
        print(f"  Images with ships: {images_with_ships}")
        print(f"  Images without ships: {images_without_ships}")

    def rle_decode(
        self,
        rle_mask: str,
        shape: Tuple[int, int] = (768, 768)
    ) -> np.ndarray:
        """
        Decode Run-Length Encoded (RLE) mask string to binary mask array.

        RLE format explanation:
            - The mask is encoded as "start1 length1 start2 length2 ..."
            - Start positions are 1-indexed in the original format
            - Pixels are in column-major (Fortran) order

        Example RLE string:
            "1 3 10 5" means:
                - 3 pixels starting at position 1 are ship pixels
                - 5 pixels starting at position 10 are ship pixels

        Args:
            rle_mask: RLE encoded string (e.g., "1 3 10 5").
                Empty string or NaN means no ship in the image.
            shape: Shape of the output mask as (height, width).
                Default is (768, 768) which is the original image size.

        Returns:
            Binary mask as numpy array with shape (height, width).
            Values are 0 (background) or 1 (ship).

        Raises:
            ValueError: If RLE string format is invalid.
        """
        # Handle empty masks (no ship present)
        if pd.isna(rle_mask) or rle_mask == '':
            return np.zeros(shape, dtype=np.uint8)

        try:
            # Parse RLE string: split into alternating start positions and lengths
            s = rle_mask.split()
            starts = np.asarray(s[0::2], dtype=int)
            lengths = np.asarray(s[1::2], dtype=int)
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid RLE mask format: {rle_mask}") from e

        # Convert from 1-indexed to 0-indexed
        starts -= 1

        # Create flattened mask (will reshape later)
        mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)

        # Fill in the mask pixels for each run
        for start, length in zip(starts, lengths):
            mask[start:start + length] = 1

        # IMPORTANT: Reshape using Fortran order (column-major)
        # The RLE encoding uses column-major order, not row-major
        return mask.reshape(shape, order='F')

    def combine_masks(
        self,
        rle_list: List[str],
        original_shape: Tuple[int, int] = (768, 768)
    ) -> np.ndarray:
        """
        Combine multiple RLE masks into a single binary mask.

        Since one image can contain multiple ships, we need to combine all
        individual ship masks into one unified binary mask for segmentation.

        Note:
            This method decodes RLE masks at their original resolution (768x768)
            and then resizes to self.img_size. This preserves mask quality better
            than decoding at target size directly.

        Args:
            rle_list: List of RLE encoded strings for one image.
                Each string represents one ship instance.
            original_shape: Original mask shape as (height, width).
                Default is (768, 768) which is the original Kaggle image size.

        Returns:
            Combined binary mask where any ship pixel is marked as 1.
            Shape is (height, width) with dtype uint8.
        """
        # Initialize empty mask at original resolution
        combined_mask = np.zeros(original_shape, dtype=np.uint8)

        # Decode each RLE mask and combine using maximum (logical OR)
        for rle in rle_list:
            if pd.notna(rle):
                mask = self.rle_decode(rle, original_shape)
                combined_mask = np.maximum(combined_mask, mask)

        return combined_mask

    def _load_and_preprocess_image(
        self,
        image_path: str,
        mask: np.ndarray
    ) -> Tuple[tf.Tensor, tf.Tensor]:

        """
        Load and preprocess a single image and its corresponding mask.

        This function is called by the tf.data.Dataset pipeline for each sample.

        Processing steps for image:
            1. Read JPEG file from disk
            2. Decode to RGB tensor (3 channels)
            3. Resize to target size using bilinear interpolation
            4. Normalize pixel values from [0, 255] to [0, 1] range

        Processing steps for mask:
            1. Add channel dimension (H, W) -> (H, W, 1)
            2. Resize to target size using nearest neighbor interpolation
            3. Convert to float32 for consistency with image

        Args:
            image_path: Full path to the image file (string).
            mask: Pre-decoded binary mask as numpy array with shape (H, W).

        Returns:
            Tuple of (image_tensor, mask_tensor) where:
                - image_tensor: shape (img_size[0], img_size[1], 3),
                  dtype float32, range [0, 1]
                - mask_tensor: shape (img_size[0], img_size[1], 1),
                  dtype float32, values {0, 1}
        """
        # Load and decode image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)

        # Resize image using bilinear interpolation (default)
        image = tf.image.resize(image, self.img_size)

        # Normalize to [0, 1] range
        image = tf.cast(image, tf.float32) / 255.0

        mask = tf.convert_to_tensor(mask)
        mask = tf.squeeze(mask)
        # Prepare mask
        # Add channel dimension: (H, W) -> (H, W, 1)
        mask = tf.reshape(mask, [768, 768, 1])

        # Resize mask using nearest neighbor interpolation
        # CRITICAL: Use 'nearest' method for binary masks
        mask = tf.image.resize(mask, self.img_size, method='nearest')

        # Convert to float32
        mask = tf.cast(mask, tf.float32)

        return image, mask

    def create_dataset(
        self,
        batch_size: int = 8,
        shuffle: bool = True,
        balance_classes: bool = True,
        df: Optional[pd.DataFrame] = None
    ) -> tf.data.Dataset:
        '''
         Creates a Balanced mini-batch
         - The batch contains the same amount of images with and without ships (50-50%)
         - Merges the ship masks for the specific images
        '''
        df = self.df_grouped.copy() if df is None else df.copy()

        if balance_classes:
            ships = df[df['has_ship']]
            no_ships = df[~df['has_ship']]
            no_ships = no_ships.sample(n=len(ships), random_state=self.seed)
            df = pd.concat([ships, no_ships]).sample(frac=1, random_state=self.seed)

        # 1. Start with simple tensors (just strings)
        image_paths = [str(self.train_image_dir / rid) for rid in df['ImageId']]
        # Join RLEs with a pipe to keep it a simple string
        rle_strings = [('|||'.join(r) if len(r) > 0 else '') for r in df['EncodedPixels']]

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, rle_strings))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1024)

        # 2. Optimization: Map with a smaller footprint
        # Use a wrapper to handle the decoding and resizing in one call
        def _parse_function(img_path, rle_str):
            img_path = img_path.numpy().decode('utf-8')
            rle_str = rle_str.numpy().decode('utf-8')
            
            # Load Image
            img = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, self.img_size) / 255.0
            
            # Decode Mask directly to smaller size if possible, or decode then resize
            if rle_str == '':
                mask = np.zeros((768, 768), dtype=np.uint8)
            else:
                mask = self.combine_masks(rle_str.split('|||'))
            
            mask = np.expand_dims(mask, axis=-1)
            # Resize mask to save memory in the prefetch buffer
            mask = tf.image.resize(mask, self.img_size, method='nearest')

            mask = tf.cast(mask, tf.float32)
            
            return img, mask

        dataset = dataset.map(
            lambda p, r: tf.py_function(_parse_function, [p, r], [tf.float32, tf.float32]),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Set shapes explicitly for the Keras model
        dataset = dataset.map(lambda i, m: (
            tf.ensure_shape(i, [self.img_size[0], self.img_size[1], 3]),
            tf.ensure_shape(m, [self.img_size[0], self.img_size[1], 1])
        ))

        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def train_val_split(self, val_split=0.2, batch_size=8):
        df = self.df_grouped.copy()
        train_df, val_df = train_test_split(
            df, test_size=val_split, stratify=df['has_ship'], random_state=self.seed
        )
        '''
        Creates the validation and train datasets with the given ratio
        '''
        train_ds = self.create_dataset(batch_size=batch_size, shuffle=True, balance_classes=True, df=train_df)
        val_ds = self.create_dataset(batch_size=batch_size, shuffle=False, balance_classes=False, df=val_df)
        
        return train_ds, val_ds


def main() -> None:
    """
    Example usage of the ShipDatasetLoader class.

    This demonstrates how to:
        1. Initialize the loader with your data paths
        2. Create train/validation split
        3. Inspect the dataset structure
    """

    # Initialize the loader
    # YOU NEED TO SPECIFY THESE PATHS:
    # - train_image_dir: folder containing the .jpg image files
    # - masks_csv_path: path to the CSV file with RLE encoded masks
    loader = ShipDatasetLoader(
        train_image_dir=TRAIN_IMAGES_DIR,
        masks_csv_path=TRAIN_METADATA_CSV,
        img_size=IMG_SIZE,
        seed=RANDOM_SEED
    )

    # Create train and validation datasets
    train_ds, val_ds = loader.train_val_split(
        val_split=VALIDATION_SPLIT,
        batch_size=BATCH_SIZE
    )

    # Test: inspect the first batch
    print("\nTesting dataset pipeline:")
    for images, masks in train_ds.take(1):
        print(f"Batch shapes: images={images.shape}, masks={masks.shape}")
        print(f"Image value range: [{images.numpy().min():.2f}, {images.numpy().max():.2f}]")
        print(f"Mask unique values: {np.unique(masks.numpy())}")
        print("Dataset pipeline test successful!")


if __name__ == "__main__":
    main()