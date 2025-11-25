import tensorflow as tf
from tensorflow.keras.utils import Sequence
import config
from data_loading import DataLoader
import numpy as np

# Optional fast ops via OpenCV; fall back when not available
try:
    import cv2
    HAVE_CV2 = True
except Exception:
    cv2 = None
    HAVE_CV2 = False

class ShipDataGenerator(Sequence):
    """
    Keras Sequence for segmentation tasks.
    Loads images and masks on-the-fly, applies resize, normalization, augmentation.
    """

    def __init__(self, image_ids, data_loader, batch_size=config.BATCH_SIZE,
                 img_size=config.IMG_SIZE, shuffle=True, augment=False):
        self.image_ids = image_ids
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.augment = augment
        # RNG seeded for reproducibility; small change but keeps behavior stable
        self.rng = np.random.default_rng(config.RANDOM_SEED if hasattr(config, 'RANDOM_SEED') else None)
        self.on_epoch_end()

    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(len(self.image_ids) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        batch_ids = [self.image_ids[i] for i in self.indices[index*self.batch_size:(index+1)*self.batch_size]]
        X, y = self._generate_batch(batch_ids)
        return X, y

    def on_epoch_end(self):
        """Shuffle after each epoch"""
        self.indices = np.arange(len(self.image_ids))
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def _generate_batch(self, batch_ids):
        """Load and preprocess a batch"""
        batch_images, batch_masks = [], []
        for image_id in batch_ids:
            img, mask = self.data_loader.get_image_mask(image_id)
            img = self._resize(img)
            mask = self._resize(mask, is_mask=True)
            if self.augment:
                img, mask = self._augment(img, mask)
            img = self._normalize(img)
            batch_images.append(img)
            batch_masks.append(mask[..., np.newaxis])  # add channel dim

        return np.array(batch_images, dtype=np.float32), np.array(batch_masks, dtype=np.float32)

    def _resize(self, img, is_mask=False):
        # Prefer OpenCV resize for speed when available, otherwise use tf.image
        h, w = self.img_size
        if HAVE_CV2:
            # Handle masks (2D) and images (3D)
            if is_mask:
                # ensure uint8
                img_in = img.astype(np.uint8)
                resized = cv2.resize(img_in, (w, h), interpolation=cv2.INTER_NEAREST)
                return resized
            else:
                img_in = img.astype(np.uint8)
                resized = cv2.resize(img_in, (w, h), interpolation=cv2.INTER_LINEAR)
                return resized
        else:
            method = tf.image.ResizeMethod.NEAREST_NEIGHBOR if is_mask else tf.image.ResizeMethod.BILINEAR
            if img.ndim == 2:
                img = img[..., np.newaxis]
            resized = tf.image.resize(img, self.img_size, method=method)
            if is_mask:
                resized = tf.squeeze(resized, axis=-1)
            return resized.numpy()

    def _normalize(self, img):
        return img / 255.0

    def _augment(self, img, mask):
        # Implement augmentations with numpy/OpenCV where possible (no TF <-> numpy roundtrips)
        # Random horizontal flip
        if config.HORIZONTAL_FLIP and self.rng.random() > 0.5:
            img = np.fliplr(img)
            mask = np.fliplr(mask)
        # Random vertical flip
        if config.VERTICAL_FLIP and self.rng.random() > 0.5:
            img = np.flipud(img)
            mask = np.flipud(mask)
        # Random brightness: scale image in HSV or simply scale RGB
        if config.BRIGHTNESS_RANGE:
            low, high = config.BRIGHTNESS_RANGE
            factor = self.rng.uniform(low, high)
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return img, mask


if __name__ == "__main__":
    # 1. Betöltjük a metadata-t
    loader = DataLoader()
    loader.load_metadata()

    # 2. Kiválasztunk néhány image_id-t
    all_ids = loader.df['ImageId'].unique()
    train_ids = all_ids[:100]  # példa

    # 3. Data generator létrehozása
    train_gen = ShipDataGenerator(train_ids, data_loader=loader, batch_size=8, augment=True)

    # 4. Teszt batch
    X, y = train_gen[0]
    print(X.shape, y.shape)  # (8, 240,240,3), (8,240,240,1)


def create_tf_dataset(image_ids, data_loader, batch_size=config.BATCH_SIZE,
                      img_size=config.IMG_SIZE, shuffle=True, augment=False):
    """Create a tf.data.Dataset that loads images and masks via the provided DataLoader.

    This uses `tf.numpy_function` to call the DataLoader (so it benefits from its
    in-memory/disk cache). The function then sets shapes, applies normalization and
    optional augmentations using TF ops for parallelism.
    """
    AUTOTUNE = tf.data.AUTOTUNE

    # Ensure image_ids is a list (tf.data handles lists/numpy arrays)
    ds = tf.data.Dataset.from_tensor_slices(list(image_ids))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(image_ids))

    def _py_load(image_id_bytes):
        # image_id_bytes is a bytes object from TF; decode to str
        image_id = image_id_bytes.decode('utf-8') if isinstance(image_id_bytes, (bytes, bytearray)) else str(image_id_bytes)
        img, mask = data_loader.get_image_mask(image_id)
        # Resize using OpenCV if available, otherwise use numpy / PIL fallback
        h, w = img_size
        try:
            if HAVE_CV2:
                img_resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
                mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                # fallback: use tf.image (but we're in numpy context), so use PIL/np
                from PIL import Image
                img_resized = np.array(Image.fromarray(img).resize((w, h), resample=Image.BILINEAR))
                mask_resized = np.array(Image.fromarray(mask).resize((w, h), resample=Image.NEAREST))
        except Exception:
            # fall back to simple numpy resize (very slow) to keep robustness
            img_resized = np.array(Image.fromarray(img).resize((w, h), resample=Image.BILINEAR))
            mask_resized = np.array(Image.fromarray(mask).resize((w, h), resample=Image.NEAREST))

        # Ensure dtypes
        img_resized = img_resized.astype(np.float32)
        mask_resized = mask_resized.astype(np.float32)
        mask_resized = np.expand_dims(mask_resized, axis=-1)

        return img_resized, mask_resized

    def _tf_wrap(image_id):
        img, mask = tf.numpy_function(_py_load, [image_id], [tf.float32, tf.float32])
        # Set static shapes so downstream ops know dimensions
        img.set_shape((img_size[0], img_size[1], config.IMG_CHANNELS))
        mask.set_shape((img_size[0], img_size[1], 1))
        return img, mask

    ds = ds.map(_tf_wrap, num_parallel_calls=AUTOTUNE)

    # Normalize to [0,1]
    ds = ds.map(lambda i, m: (i / 255.0, m), num_parallel_calls=AUTOTUNE)

    # Apply augmentations using TF ops (fast and parallel)
    if augment:
        def _augment_tf(image, mask):
            # Concatenate to apply geometric transforms identically
            concat = tf.concat([image, mask], axis=-1)
            concat = tf.image.random_flip_left_right(concat)
            # random 90-degree rotations
            k = tf.random.uniform([], 0, 4, dtype=tf.int32)
            concat = tf.image.rot90(concat, k)
            image_aug = concat[..., :config.IMG_CHANNELS]
            mask_aug = concat[..., config.IMG_CHANNELS:]
            # color jitter on image only
            image_aug = tf.image.random_brightness(image_aug, 0.2)
            image_aug = tf.image.random_contrast(image_aug, 0.8, 1.2)
            image_aug = tf.clip_by_value(image_aug, 0.0, 1.0)
            return image_aug, mask_aug

        ds = ds.map(_augment_tf, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds
