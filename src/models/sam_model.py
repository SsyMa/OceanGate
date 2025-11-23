"""SAM-based model wrapper for Airbus ship detection.

This file provides a lightweight PyTorch wrapper around Facebook Research's
Segment Anything (SAM) model for use as a segmentation backbone/decoder in
the Kaggle Airbus Ship Detection task.

Notes:
- This module intentionally avoids hard-coding a specific factory function
  from the `segment_anything` package. If `segment_anything` is available in
  the environment, helper constructors will try to use its public helpers
  (`sam_model_registry`, `SamPredictor`, `SamAutomaticMaskGenerator`). If not
  available, users can still instantiate `SAMShipDetector` by passing an
  already-constructed SAM model object.
- The wrapper focuses on model architecture and inference contract (inputs/
  outputs); training loops, dataset wiring, and checkpoint downloads are
  intentionally outside the scope of this file.

Example usage:

    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry["vit_h"](checkpoint="/path/to/checkpoint.pth")
    detector = SAMShipDetector(sam)
    masks, scores = detector(images, prompts=prompts)

"""
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn


class SAMShipDetector(nn.Module):
    """Wraps a Segment Anything (SAM) model for ship segmentation.

    The wrapper standardizes the inputs and outputs expected for the
    airplane/ship detection pipeline. The class does not attempt to
    reimplement SAM internals; instead it delegates to a passed-in SAM
    model instance (or uses helpers from the `segment_anything` package if
    available).

    Forward contract:
        - images: Tensor of shape (B, C, H, W), float, in whatever range the
          underlying SAM model expects (usually [0, 1] or normalized).
        - prompts: optional dictionary with keys like "points", "boxes",
          "masks" representing prompts for SAM. The wrapper will pass these
          to the predictor if available.

    Returns a tuple (masks, scores, extra) where:
        - masks: List[Tensor] or Tensor of predicted binary masks per image.
        - scores: List[float] confidence scores per mask.
        - extra: dict of additional outputs coming from SAM (e.g., iou_scores)

    The exact types for masks/scores mirror the behavior of the chosen SAM
    implementation (the wrapper does not convert them to a specific format
    to preserve flexibility).
    """

    def __init__(self, sam: Optional[nn.Module] = None, use_predictor: bool = True):
        """Create a SAMShipDetector.

        Args:
            sam: An instantiated SAM model object (from `segment_anything`).
                 If `None`, helper constructors in this module can be used to
                 create one via `build_sam_from_registry()`.
            use_predictor: If True and `segment_anything.SamPredictor` is
                 available, the wrapper will build a predictor for prompt
                 handling and mask generation.
        """
        super().__init__()
        self.sam = sam
        self.predictor = None
        self._has_segment_anything = False

        try:
            # Try to import optional helpers from the `segment_anything` package.
            # These imports are optional so this file can remain importable even
            # when the package isn't installed.
            from segment_anything import SamPredictor  # type: ignore

            self._has_segment_anything = True
            if use_predictor and sam is not None:
                self.predictor = SamPredictor(sam)
        except Exception:
            # If the package or SamPredictor isn't available, we fall back to
            # delegating directly to the provided `sam` object (if any).
            self._has_segment_anything = False

    def forward(self, images: torch.Tensor, prompts: Optional[Dict[str, Any]] = None) -> Tuple[Any, Any, Dict[str, Any]]:
        """Run inference through SAM with optional prompts.

        Args:
            images: Tensor[B, C, H, W]
            prompts: Optional dict containing keys compatible with the chosen
                predictor; common keys include `points` (Nx2 coords), `boxes`
                (x1,y1,x2,y2), `masks` (binary mask prompts).

        Returns:
            masks, scores, extra
        """
        if self.predictor is not None:
            # When a SamPredictor is available, use it to set the image and
            # handle prompt-to-mask conversion. The predictor API typically
            # exposes `set_image` and `predict` methods. We defensively call
            # them and return what we get back.
            try:
                # SamPredictor expects numpy images in many examples; but some
                # builds accept torch tensors. We'll try tensor path first,
                # otherwise convert to numpy in CPU memory.
                if isinstance(images, torch.Tensor):
                    img_for_predictor = images
                else:
                    img_for_predictor = images

                self.predictor.set_image(img_for_predictor)

                # Compose arguments for predictor.predict from provided
                # prompts. We deliberately do not enforce a strict schema so
                # callers can pass what their SAM version expects.
                predict_kwargs = {}
                if prompts is not None:
                    predict_kwargs.update(prompts)

                out = self.predictor.predict(**predict_kwargs)

                # Many SAM predictors return a tuple like (masks, scores, logits)
                # or a dict. We normalize into (masks, scores, extra).
                if isinstance(out, tuple) and len(out) >= 2:
                    masks, scores = out[0], out[1]
                    extra = {"raw_output": out[2:]} if len(out) > 2 else {}
                elif isinstance(out, dict):
                    masks = out.get("masks")
                    scores = out.get("scores")
                    extra = {k: v for k, v in out.items() if k not in ("masks", "scores")}
                else:
                    masks = out
                    scores = None
                    extra = {}

                return masks, scores, extra
            except Exception as exc:
                # If predictor usage fails, try to fall back to direct SAM
                # forwarding below.
                # Re-raise only if there is no fallback available.
                if self.sam is None:
                    raise

        # Fallback: delegate to `self.sam` directly. Different SAM
        # implementations use different forward signatures; we try a couple
        # of common patterns while remaining non-destructive.
        if self.sam is None:
            raise RuntimeError("No SAM model or predictor available. Install 'segment_anything' or pass a SAM instance to SAMShipDetector.")

        # Typical sam model may implement a `forward` that returns features or
        # mask logits. We pass images and prompts where available.
        try:
            if prompts is None:
                out = self.sam(images)
            else:
                out = self.sam(images, **prompts)

            # Normalize output into masks, scores, extra dict when possible.
            if isinstance(out, tuple) and len(out) >= 2:
                masks, scores = out[0], out[1]
                extra = {"raw_output": out[2:]} if len(out) > 2 else {}
            elif isinstance(out, dict):
                masks = out.get("masks")
                scores = out.get("scores")
                extra = {k: v for k, v in out.items() if k not in ("masks", "scores")}
            else:
                masks = out
                scores = None
                extra = {}

            return masks, scores, extra
        except Exception as exc:
            # Surface a helpful error for developers integrating this wrapper.
            raise RuntimeError("Failed to run SAM model. Ensure the provided `sam` instance implements a callable forward or install 'segment_anything'.") from exc


def build_sam_from_registry(model_type: str, checkpoint: str):
    """Helper to construct a SAM model using `segment_anything`'s registry.

    This function tries to import `sam_model_registry` from
    `segment_anything` and construct a model. If the package is not
    available, it raises an ImportError with guidance.

    Args:
        model_type: name of the model type in the registry (e.g., 'vit_h').
        checkpoint: path to a SAM checkpoint file.

    Returns:
        An instantiated SAM model object.
    """
    try:
        from segment_anything import sam_model_registry  # type: ignore

        if model_type not in sam_model_registry:
            raise KeyError(f"Model type '{model_type}' not found in sam_model_registry. Available: {list(sam_model_registry.keys())}")

        sam_ctor = sam_model_registry[model_type]
        sam = sam_ctor(checkpoint=checkpoint)
        return sam
    except Exception as exc:
        raise ImportError(
            "Could not build SAM from registry. Ensure 'segment_anything' is installed and a valid checkpoint path is provided."
        ) from exc


def build_automatic_mask_generator(sam_model, **kwargs):
    """Constructs a `SamAutomaticMaskGenerator` if available.

    This helper returns a mask generator instance that can produce
    automatic proposals for an image without prompts. If the class is not
    available it raises ImportError.
    """
    try:
        from segment_anything import SamAutomaticMaskGenerator  # type: ignore

        return SamAutomaticMaskGenerator(sam_model, **kwargs)
    except Exception as exc:
        raise ImportError(
            "SamAutomaticMaskGenerator is not available. Install 'segment_anything' to use automatic mask generation."
        ) from exc
