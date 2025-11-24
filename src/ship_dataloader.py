import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cv2

# Constants
IMG_SIZE = 768  # Airbus native size
RESIZE_TO = 1024 # SAM standard input size (ViT expects this)
BATCH_SIZE = 2

def rle_decode(mask_rle, shape=(768, 768)):
    """Decodes RLE mask string to binary grid."""
    if not isinstance(mask_rle, str):
        return np.zeros(shape, dtype=np.uint8)
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T # Transpose required for Airbus dataset

def get_box_with_perturbation(mask, noise_scale=5):
    """
    Calculates bounding box from mask and adds random noise (jitter).
    This prevents the model from overfitting to 'perfect' boxes.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows):
        return np.array([0, 0, 1, 1], dtype="float32") # Dummy
        
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Add noise (simulating imperfect detection)
    x1 = max(0, cmin - np.random.randint(0, noise_scale))
    y1 = max(0, rmin - np.random.randint(0, noise_scale))
    x2 = min(IMG_SIZE, cmax + np.random.randint(0, noise_scale))
    y2 = min(IMG_SIZE, rmax + np.random.randint(0, noise_scale))
    
    # SAM expects [x1, y1, x2, y2] format
    return np.array([x1, y1, x2, y2], dtype="float32")

def generator_fn(df, image_dir):
    """
    Python generator yielding one sample at a time:
    (Inputs Dictionary, Targets Dictionary)
    """
    for _, row in df.iterrows():
        img_id = row['ImageId']
        rle = row['EncodedPixels']
        
        # 1. Load Image
        img_path = os.path.join(image_dir, img_id)
        try:
            # Read standard RGB
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            continue # Skip corrupt images

        # 2. Load Mask & Create Box
        mask = rle_decode(rle)
        
        # Skip empty images (Background) for fine-tuning
        # We want to teach the model "Ship features", not "Water features"
        if np.sum(mask) == 0:
            continue
            
        box = get_box_with_perturbation(mask)

        # 3. Format Inputs
        # SAM Inputs: 
        #   images: [H, W, 3]
        #   boxes:  [1, 2, 2] -> (1 box, 2 points (TL/BR), 2 coords (x/y))
        #   labels: [1]       -> 1 means "foreground/box"
        
        # Reshape box to KerasCV SAM format: (1, 2, 2)
        # The prompt expects points: TopLeft, BottomRight
        box_reshaped = np.array([
            [box[0], box[1]], # x1, y1
            [box[2], box[3]]  # x2, y2
        ]).reshape(1, 2, 2)

        inputs = {
            "images": img,
            "boxes": box_reshaped,
            "labels": np.array([1], dtype="int32") 
        }

        # 4. Format Targets
        # SAM Targets:
        #   masks: [1, H, W] (Channel First for some implementations, check KerasCV docs)
        #   Usually Keras prefers [H, W, 1]. Let's verify the specific preset expectations.
        #   Standard KerasCV SAM expects masks aligned with image spatial dims.
        
        targets = {
            "masks": np.expand_dims(mask, axis=-1), # (768, 768, 1)
            "iou_pred": np.array([1.0]) # We are training on GT, so IoU is 1.0
        }

        yield inputs, targets

def create_dataset(csv_path, image_dir):
    df = pd.read_csv(csv_path)
    # Filter out images with no ships to speed up training (optional but recommended)
    df = df.dropna(subset=['EncodedPixels']) 
    
    # Define output signature
    output_signature = (
        {
            "images": tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.uint8),
            "boxes": tf.TensorSpec(shape=(1, 2, 2), dtype=tf.float32),
            "labels": tf.TensorSpec(shape=(1,), dtype=tf.int32),
        },
        {
            "masks": tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 1), dtype=tf.uint8),
            "iou_pred": tf.TensorSpec(shape=(1,), dtype=tf.float32),
        }
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: generator_fn(df, image_dir),
        output_signature=output_signature
    )
    
    # 5. Preprocessing Pipeline (Resize and Normalize)
    def preprocess_map(inputs, targets):
        # Resize Image
        inputs["images"] = tf.image.resize(inputs["images"], (RESIZE_TO, RESIZE_TO))
        inputs["images"] = tf.cast(inputs["images"], tf.float32)
        
        # Resize Mask
        targets["masks"] = tf.image.resize(targets["masks"], (RESIZE_TO, RESIZE_TO), method="nearest")
        targets["masks"] = tf.cast(targets["masks"], tf.float32)
        
        # Scale Boxes to new resolution
        scale = RESIZE_TO / IMG_SIZE
        inputs["boxes"] = inputs["boxes"] * scale
        
        return inputs, targets

    dataset = dataset.map(preprocess_map, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset