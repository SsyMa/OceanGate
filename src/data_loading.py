import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import config

# Optional fast image ops via OpenCV; fall back to PIL if not available
try:
    import cv2
    HAVE_CV2 = True
except Exception:
    cv2 = None
    HAVE_CV2 = False

class DataLoader:
    """
    Data loader for binary segmentation (e.g., ship detection)
    Loads images and decodes RLE masks.
    """

    def __init__(self):
        self.df = None
        self.image_dir = config.TRAIN_IMAGES_DIR
        self.rle_map = None

    def load_metadata(self):
        """Load CSV metadata and prepare dataset"""
        self.df = pd.read_csv(config.TRAIN_METADATA_CSV)
        # Add a column indicating if image has a ship
        self.df['has_ship'] = self.df['EncodedPixels'].notnull()
        # Build a mapping ImageId -> list of RLEs to avoid per-image dataframe filtering
        self.rle_map = self.df.groupby('ImageId')['EncodedPixels'].apply(list).to_dict()
        return self.df

    def rle_decode(self, mask_rle, shape=(config.ORIGINAL_IMG_HEIGHT, config.ORIGINAL_IMG_WIDTH)):
        """Decode run-length encoded mask to binary 2D array"""
        if pd.isna(mask_rle):
            return np.zeros(shape, dtype=np.uint8)

        s = mask_rle.split()
        starts, lengths = np.array(s[0::2], dtype=int) - 1, np.array(s[1::2], dtype=int)
        ends = starts + lengths
        mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for start, end in zip(starts, ends):
            mask[start:end] = 1
        return mask.reshape(shape).T

    def get_image_mask(self, image_id):
        """Load image and combined mask for a given image_id"""
        img_path = os.path.join(self.image_dir, image_id)
        # Prefer OpenCV for speed when available
        if HAVE_CV2:
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if image is None:
                # fallback to PIL if cv2 fails to read
                image = Image.open(img_path).convert('RGB')
                image = np.array(image)
            else:
                # convert BGR -> RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)

        # Combine all masks for this image using precomputed rle_map
        mask = np.zeros((config.ORIGINAL_IMG_HEIGHT, config.ORIGINAL_IMG_WIDTH), dtype=np.uint8)
        if self.rle_map is None:
            # backward-compatible: fall back to dataframe filtering
            rles = self.df[self.df['ImageId'] == image_id]['EncodedPixels']
        else:
            rles = self.rle_map.get(image_id, [])

        for rle in rles:
            if pd.notna(rle):
                mask += self.rle_decode(rle)
        mask = np.clip(mask, 0, 1)
        return image, mask

    def visualize_sample(self, image_id):
        """Quick visualization of an image and its mask"""
        image, mask = self.get_image_mask(image_id)
        plt.figure(figsize=(12,5))
        plt.subplot(1,3,1)
        plt.imshow(image); plt.title("Image"); plt.axis('off')
        plt.subplot(1,3,2)
        plt.imshow(mask, cmap='gray'); plt.title("Mask"); plt.axis('off')
        plt.subplot(1,3,3)
        plt.imshow(image)
        plt.imshow(mask, cmap='Reds', alpha=0.4)
        plt.title("Overlay"); plt.axis('off')
        plt.show()
        return image, mask
