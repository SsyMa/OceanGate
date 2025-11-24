import tensorflow as tf
from ship_sam_model import get_ship_sam_model, compile_sam_model
from ship_dataloader import create_dataset

# Paths
CSV_PATH = "./train_ship_segmentations_v2.csv"
IMG_DIR = "./train_v2/"

def main():
    print("Initializing Dataset...")
    # Create the TF Data Pipeline
    train_ds = create_dataset(CSV_PATH, IMG_DIR)
    
    # Check shape of one batch (debugging)
    for inp, targ in train_ds.take(1):
        print(f"Image Batch Shape: {inp['images'].shape}")
        print(f"Box Batch Shape: {inp['boxes'].shape}")
        print(f"Mask Target Shape: {targ['masks'].shape}")

    print("Building Model...")
    # Initialize Model (Freezing backbone)
    model = get_ship_sam_model(input_shape=(1024, 1024, 3), freeze_backbone=True)
    
    # Compile
    model = compile_sam_model(model, learning_rate=1e-4)
    
    # Callbacks
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        "sam_ship_best.keras", 
        save_best_only=True, 
        monitor="loss"
    )
    
    print("Starting Training...")
    # Fit
    model.fit(
        train_ds,
        epochs=5, # Start small, SAM fine-tunes quickly
        callbacks=[checkpoint_cb]
    )

if __name__ == "__main__":
    main()