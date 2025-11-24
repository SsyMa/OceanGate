import keras
import keras_cv
import tensorflow as tf
import numpy as np

# Use 'sam_base_sa1b' for a balance of speed/accuracy. 
# Options: 'sam_huge_sa1b', 'sam_large_sa1b', 'sam_base_sa1b'
PRESET_MODEL = "sam_base_sa1b"

def get_ship_sam_model(input_shape=(768, 768, 3), freeze_backbone=True):
    """
    Initializes the Segment Anything Model (SAM) for fine-tuning.
    
    Args:
        input_shape: The resize shape (Airbus images are 768x768).
        freeze_backbone: If True, freezes the heavy Image Encoder (ViT).
                         Recommended for 99% of use cases to prevent OOM errors.
    """
    # 1. Load the pre-trained SAM model from KerasCV
    backbone = keras_cv.models.SegmentAnythingModel.from_preset(
        PRESET_MODEL,
        dtype="float32"  # Use mixed_float16 if you have a modern GPU
    )

    # 2. Configuration for Training
    if freeze_backbone:
        # We only want to train the Mask Decoder (the lightweight part)
        backbone.backbone.trainable = False
        backbone.prompt_encoder.trainable = False
        backbone.mask_decoder.trainable = True
    
    # 3. Wrap in a functional Keras model for custom input handling
    # SAM expects a dictionary of inputs. We wrap this to make .fit() easier later.
    inputs = {
        "images": keras.Input(shape=input_shape, name="images"),
        "boxes": keras.Input(shape=(None, 2, 2), name="boxes"), # Shape: (Batch, Num_Boxes, 2 points, 2 coords)
        "labels": keras.Input(shape=(None,), dtype="int32", name="labels") # Box presence flags
    }
    
    # Forward pass
    outputs = backbone(inputs)
    
    # Construct the final model
    # Output: {"masks": ..., "iou_pred": ...}
    model = keras.Model(inputs=inputs, outputs=outputs, name="sam_ship_segmentor")
    
    return model

def compile_sam_model(model, learning_rate=1e-4):
    """
    Compiles the model with a loss function suitable for segmentation masks.
    SAM outputs logits, so we need a loss that handles that.
    """
    optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.004)
    
    # Custom loss dictionary because SAM outputs a dictionary
    losses = {
        "masks": keras_cv.losses.FocalLoss(from_logits=True, reduction="sum_over_batch_size"),
        "iou_pred": keras.losses.MeanSquaredError() # Auxiliary loss for IoU prediction
    }
    
    # Weights for the losses (Mask accuracy is priority)
    loss_weights = {"masks": 1.0, "iou_pred": 0.05}
    
    model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)
    return model

# --- Helper Functions for Data Pipeline ---

def rle_decode(mask_rle, shape=(768, 768)):
    """
    Decodes the Airbus RLE (Run Length Encoding) string into a binary mask.
    Required for processing the Kaggle CSV.
    """
    if not isinstance(mask_rle, str):
        return np.zeros(shape, dtype=np.uint8)
        
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Transpose needed for Airbus dataset

def convert_mask_to_box_prompt(mask):
    """
    Generates a bounding box from a binary mask.
    During training, we 'teach' SAM: "If I give you this box, give me the ship mask."
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows):
        # Empty mask (no ship), return dummy box
        return np.array([[0, 0, 1, 1]], dtype="float32") 
        
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Add slight noise/jitter to box to make model robust (optional but recommended)
    # Format: [x1, y1, x2, y2]
    return np.array([[cmin, rmin, cmax, rmax]], dtype="float32")