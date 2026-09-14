"""Point/box/mask-based *alpha matting* using SAM2Matting.

SAM2Matting (FudanCVL) = SAM2 tracker + dedicated matting heads. Unlike the
other MaskProcessor backends it returns a **soft alpha matte** in ``[0, 1]``
(hair strands, motion blur, translucency) instead of a hard 0/1 mask.

DeepFaceLab's XSeg pipeline is strictly binary (``DynamicSampleGenerator``
thresholds every mask at 0.5 before training), so callers must convert the
alpha with :func:`MaskProcessor.core.mask_ops.alpha_to_binary` before it is
written to a DFLJPG. This class deliberately returns the raw alpha so the UI
can let the user pick the threshold interactively.

Usage::

    predictor = SAM2MattingPredictor(model_name="sam2.1_tiny")
    predictor.load_image(bgr_image)
    alpha = predictor.predict([(x, y, 1)])            # points -> alpha
    alpha = predictor.predict_with_box((x1, y1, x2, y2))
    alpha = predictor.refine(binary_mask)             # existing mask -> alpha
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import torch

from xlib.models.sam2.build_sam import SAM2MATTING_MODELS, build_sam2matting
from xlib.models.sam2.sam2matting_image_predictor import SAM2MattingImagePredictor

PROJECT_ROOT = Path(__file__).parent.parent.parent
SAM2_PKG_DIR = PROJECT_ROOT / "xlib" / "models" / "sam2"
SAM2MATTING_CKPT_DIR = SAM2_PKG_DIR / "checkpoints"

HF_CHECKPOINT_URL = "https://huggingface.co/FudanCVL/SAM2Matting/tree/main/checkpoints"


class SAM2MattingPredictor:
    """Thin MaskProcessor-facing wrapper around :class:`SAM2MattingImagePredictor`."""

    MODELS = SAM2MATTING_MODELS

    def __init__(self, model_name: str = "sam2.1_tiny", device: Optional[str] = None):
        if model_name not in self.MODELS:
            raise ValueError(
                f"Unknown SAM2Matting model '{model_name}'. "
                f"Available: {', '.join(self.MODELS)}"
            )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        ckpt_name, cfg_rel = self.MODELS[model_name]
        ckpt_path = SAM2MATTING_CKPT_DIR / ckpt_name
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"SAM2Matting checkpoint not found: {ckpt_path}\n"
                f"Download '{ckpt_name}' from {HF_CHECKPOINT_URL} and place it there."
            )

        model = build_sam2matting(cfg_rel, str(ckpt_path), device=device)
        self.predictor = SAM2MattingImagePredictor(model)
        self.model_name = model_name
        self._device = device
        self._image_hw: Optional[tuple[int, int]] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _autocast(self):
        """bf16 autocast on CUDA (matches upstream inference scripts); no-op on CPU."""
        if str(self._device).startswith("cuda"):
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _require_image(self):
        if self._image_hw is None:
            raise RuntimeError("Call load_image() before predict().")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_image(self, image: np.ndarray) -> None:
        """Set image for matting.

        Args:
            image: BGR numpy array (H, W, 3) — the standard OpenCV format.
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self._image_hw = rgb.shape[:2]
        with torch.inference_mode(), self._autocast():
            self.predictor.set_image(rgb)

    def predict(self, clicks: Sequence[tuple], box: Optional[tuple] = None) -> np.ndarray:
        """Generate an alpha matte from point clicks (optionally plus a box).

        Args:
            clicks: ``(x, y, label)`` tuples; ``label=1`` foreground, ``0`` background.
            box: Optional ``(x1, y1, x2, y2)``; SAM2 encodes box + points jointly.

        Returns:
            float32 alpha in [0, 1] with shape (H, W).
        """
        self._require_image()
        point_coords = None
        point_labels = None
        if clicks:
            point_coords = np.array([(x, y) for x, y, _ in clicks], dtype=np.float32)
            point_labels = np.array([label for _, _, label in clicks], dtype=np.int32)
        box_np = np.array(box, dtype=np.float32) if box is not None else None
        if point_coords is None and box_np is None:
            raise ValueError("Provide clicks, a box, or both.")

        with torch.inference_mode(), self._autocast():
            masks, scores, low_res = self.predictor.predict_binary(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box_np,
                multimask_output=True,
            )
            # Feed the most confident SAM proposal to the matting heads.
            best = int(np.argmax(scores))
            coarse_logits = low_res[:, best : best + 1]  # (1, 1, 256, 256)
            alpha = self.predictor.predict_alpha(coarse_logits, output_hw=self._image_hw)
        return alpha

    def predict_with_box(self, box: tuple) -> np.ndarray:
        """Generate an alpha matte from a bounding box ``(x1, y1, x2, y2)``."""
        return self.predict([], box=box)

    def refine(self, mask: np.ndarray) -> np.ndarray:
        """Turn an existing (binary or soft) mask into an alpha matte.

        This is the upstream ``inference_image_sam2.py`` path: the mask is used
        as the coarse guide for the ROI detector + progressive matting heads.

        Args:
            mask: (H, W) float32 in [0, 1] or bool; any resolution. It is
                resized to the loaded image before being fed to the model.

        Returns:
            float32 alpha in [0, 1] with shape (H, W) of the loaded image.
        """
        self._require_image()
        h, w = self._image_hw
        m = mask.astype(np.float32)
        if m.ndim == 3:
            m = m[..., 0]
        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        coarse = (m > 0.5).astype(np.float32)
        if coarse.sum() == 0:
            return np.zeros((h, w), dtype=np.float32)

        with torch.inference_mode(), self._autocast():
            alpha = self.predictor.predict_alpha(coarse, output_hw=(h, w))
        return alpha
