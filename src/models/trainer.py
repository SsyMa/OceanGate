"""Simple PyTorch trainer for ship segmentation.

This trainer is intentionally minimal and framework-agnostic: it provides
an easy-to-run training loop that expects a segmentation model compatible
with the `SAMShipDetector` wrapper in this repo (see
`src/models/sam_model.py`). The trainer implements a lightweight
PyTorch `Dataset` that reads the same CSV used by the TensorFlow loader
and decodes RLE masks on the fly.

Usage (quick):

    python -m src.models.trainer --epochs 10 --batch-size 8

Notes:
- The trainer assumes the model's forward returns `(masks, scores, extra)`.
  For training we expect `masks` to be a Tensor of logits with shape
  `(B, 1, H, W)` or `(B, H, W)`. The code attempts to be defensive about
  different return types but you should verify your SAM variant's outputs.
- The trainer uses BCEWithLogitsLoss by default, which expects raw logits.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import TRAIN_IMAGES_DIR, TRAIN_METADATA_CSV, IMG_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE
from src.models.sam_model import SAMShipDetector, build_sam_from_registry


def rle_decode(rle_mask: str, shape: Tuple[int, int]) -> np.ndarray:
    """Decode a single RLE string to a binary mask (height, width).

    This mirrors the logic used by the TensorFlow loader.
    """
    if pd.isna(rle_mask) or rle_mask == "":
        return np.zeros(shape, dtype=np.uint8)

    s = rle_mask.split()
    starts = np.asarray(s[0::2], dtype=int)
    lengths = np.asarray(s[1::2], dtype=int)
    starts -= 1
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, length in zip(starts, lengths):
        mask[start:start + length] = 1
    return mask.reshape(shape, order='F')


def combine_masks(rle_list: List[str], original_shape: Tuple[int, int]) -> np.ndarray:
    combined = np.zeros(original_shape, dtype=np.uint8)
    for rle in rle_list:
        if pd.notna(rle) and rle != '':
            m = rle_decode(rle, original_shape)
            combined = np.maximum(combined, m)
    return combined


class PyTorchShipDataset(Dataset):
    """PyTorch dataset that mirrors the CSV grouping used by the TF loader.

    Each item returns `(image_tensor, mask_tensor)` where:
      - image_tensor: FloatTensor (3, H, W), values in [0, 1]
      - mask_tensor: FloatTensor (1, H, W), values 0 or 1
    """

    def __init__(
        self,
        images_dir: str = TRAIN_IMAGES_DIR,
        masks_csv: str = TRAIN_METADATA_CSV,
        img_size: Tuple[int, int] = IMG_SIZE,
        subset: Optional[int] = None,
        balance: bool = False,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.img_size = img_size

        df = pd.read_csv(masks_csv)
        df['has_ship'] = df['EncodedPixels'].notna()
        df_grouped = df.groupby('ImageId').agg({
            'EncodedPixels': lambda x: list(x.dropna()),
            'has_ship': 'max'
        }).reset_index()

        if balance:
            ships = df_grouped[df_grouped['has_ship']]
            no_ships = df_grouped[~df_grouped['has_ship']].sample(n=len(ships), random_state=42)
            df_grouped = pd.concat([ships, no_ships]).sample(frac=1, random_state=42)

        if subset:
            df_grouped = df_grouped.sample(n=min(subset, len(df_grouped)), random_state=42)

        self.samples = []
        for _, row in df_grouped.iterrows():
            image_path = str(self.images_dir / row['ImageId'])
            rle_list = row['EncodedPixels']
            self.samples.append((image_path, rle_list))

        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, rle_list = self.samples[idx]

        # Load image via PIL
        img = Image.open(img_path).convert('RGB')
        img = img.resize(self.img_size, resample=Image.BILINEAR)
        img_t = self.to_tensor(img)  # (3, H, W), float [0,1]

        # Combine and resize mask
        mask_np = combine_masks(rle_list, original_shape=(768, 768))
        mask_img = Image.fromarray((mask_np * 255).astype('uint8'))
        mask_img = mask_img.resize(self.img_size, resample=Image.NEAREST)
        mask_t = self.to_tensor(mask_img)  # (1, H, W)
        # Convert mask to binary {0,1}
        mask_t = (mask_t > 0.5).float()

        return img_t, mask_t


class Trainer:
    """Minimal training harness for segmentation models.

    The trainer expects a model that when called with images returns a tuple
    `(masks, scores, extra)` where `masks` is usable for computing loss.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        save_dir: str = 'checkpoints',
        lr: float = LEARNING_RATE,
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def prepare_loaders(self, batch_size: int = BATCH_SIZE, val_split: float = 0.15):
        dataset = PyTorchShipDataset()
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        self.val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    def _unpack_predicted_masks(self, masks) -> torch.Tensor:
        """Try to convert various SAM outputs into a tensor of logits (B,1,H,W).

        This function is defensive: SAM variants return different types.
        """
        if isinstance(masks, torch.Tensor):
            pred = masks
        elif isinstance(masks, (list, tuple)):
            # common pattern: list of masks per batch element
            try:
                pred = torch.stack([torch.as_tensor(m) for m in masks], dim=0)
            except Exception:
                pred = torch.as_tensor(np.array(masks))
        else:
            # try numpy
            pred = torch.as_tensor(np.array(masks))

        # Ensure float and on device
        pred = pred.float().to(self.device)

        # Expected shapes: (B, H, W) or (B, 1, H, W) or (N, H, W)
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)

        return pred

    def train(self, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, save_every: int = 1):
        self.prepare_loaders(batch_size=batch_size)

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            for images, masks in self.train_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)  # (B,1,H,W)

                self.optimizer.zero_grad()

                out = self.model(images)
                # model may return (masks, scores, extra)
                if isinstance(out, tuple) or isinstance(out, list):
                    pred_masks = out[0]
                elif isinstance(out, dict):
                    pred_masks = out.get('masks')
                else:
                    pred_masks = out

                if pred_masks is None:
                    raise RuntimeError('Model did not return masks usable for training.')

                pred_logits = self._unpack_predicted_masks(pred_masks)

                # Resize logits if needed to match target (simple nearest / interpolate)
                if pred_logits.shape[-2:] != masks.shape[-2:]:
                    pred_logits = nn.functional.interpolate(pred_logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)

                loss = self.criterion(pred_logits, masks)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * images.size(0)

            epoch_loss = running_loss / len(self.train_loader.dataset)
            val_loss = self.validate()

            print(f"Epoch {epoch}/{epochs}  Train Loss: {epoch_loss:.4f}  Val Loss: {val_loss:.4f}")

            if epoch % save_every == 0:
                ckpt_path = self.save_dir / f'model_epoch{epoch:03d}.pth'
                torch.save(self.model.state_dict(), ckpt_path)

    def validate(self) -> float:
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                out = self.model(images)
                if isinstance(out, (tuple, list)):
                    pred_masks = out[0]
                elif isinstance(out, dict):
                    pred_masks = out.get('masks')
                else:
                    pred_masks = out

                pred_logits = self._unpack_predicted_masks(pred_masks)

                if pred_logits.shape[-2:] != masks.shape[-2:]:
                    pred_logits = nn.functional.interpolate(pred_logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)

                loss = self.criterion(pred_logits, masks)
                running_loss += loss.item() * images.size(0)

        return running_loss / len(self.val_loader.dataset)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--model-type', type=str, default=None, help='SAM model type for registry (e.g. vit_h)')
    parser.add_argument('--sam-checkpoint', type=str, default=None, help='Path to SAM checkpoint if using registry')
    parser.add_argument('--checkpoint-out', type=str, default='checkpoints')
    args = parser.parse_args(argv)

    # Build or require a SAM model
    sam = None
    if args.model_type and args.sam_checkpoint:
        try:
            sam = build_sam_from_registry(args.model_type, args.sam_checkpoint)
        except Exception as exc:
            print(f"Failed to build SAM from registry: {exc}")
            return
    else:
        print("No SAM registry model provided. You must pass a `sam` instance to Trainer or use --model-type and --sam-checkpoint.")
        return

    detector = SAMShipDetector(sam)
    trainer = Trainer(detector, save_dir=args.checkpoint_out, lr=args.lr)
    trainer.train(epochs=args.epochs, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
