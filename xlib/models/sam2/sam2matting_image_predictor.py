"""SAM2Matting single-image predictor (ported from FudanCVL/SAM2Matting).

Two-stage pipeline on a single image:

1. **Segmentation** – the regular SAM2 prompt encoder / mask decoder turns
   point or box prompts into a binary mask plus 256x256 low-res logits.
2. **Matting** – the SAM2Matting alpha heads (``SAM2MattingBase``) take the
   image features, the normalised 1024x1024 input image and a coarse 256x256
   binary mask and regress a soft alpha matte in ``[0, 1]``.

The class keeps the upstream ``predict(img, raw_mask, mask_input)`` entry
point for compatibility, and adds the more explicit :meth:`predict_binary`
and :meth:`predict_alpha` helpers that MaskProcessor uses.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL.Image import Image

from xlib.models.sam2.modeling.sam2matting_base import SAM2MattingBase
from xlib.models.sam2.utils.transforms import SAM2Transforms

# Resolution of the coarse mask fed to the alpha heads (SAM low-res mask size).
MATTING_MASK_INPUT_SIZE = 256


class SAM2MattingImagePredictor:
    def __init__(
        self,
        sam_model: SAM2MattingBase,
        mask_threshold: float = 0.0,
        max_hole_area: float = 0.0,
        max_sprinkle_area: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = sam_model
        self._transforms = SAM2Transforms(
            resolution=self.model.image_size,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )
        self._is_image_set = False
        self._features = None
        self._orig_hw: Optional[List[Tuple[int, int]]] = None
        self._input_image: Optional[torch.Tensor] = None
        self.all_features: Optional[List[torch.Tensor]] = None
        self.mask_threshold = mask_threshold
        # Spatial sizes of the three FPN levels for a 1024 input (stride 4/8/16).
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]

    # ------------------------------------------------------------------
    # Image encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def set_image(self, image: Union[np.ndarray, Image]) -> torch.Tensor:
        """Encode an RGB image and cache its multi-scale features.

        Returns the normalised ``1x3xSxS`` tensor that was fed to the image
        encoder. The alpha heads need it again, so it is also cached on the
        instance as ``_input_image``.
        """
        self.reset_predictor()

        if isinstance(image, np.ndarray):
            self._orig_hw = [image.shape[:2]]
        elif isinstance(image, Image):
            w, h = image.size
            self._orig_hw = [(h, w)]
        else:
            raise NotImplementedError("Image format not supported")

        input_image = self._transforms(image)
        input_image = input_image[None, ...].to(self.device)

        assert (
            len(input_image.shape) == 4 and input_image.shape[1] == 3
        ), f"input_image must be of size 1x3xHxW, got {input_image.shape}"

        backbone_out = self.model.forward_image(input_image)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)

        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self.all_features = feats
        self._input_image = input_image
        self._is_image_set = True
        logging.info("SAM2Matting: image embeddings computed.")
        return input_image

    # ------------------------------------------------------------------
    # Stage 1 — prompts -> binary mask (regular SAM2 decoder)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_binary(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        normalize_coords: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
        """Run the SAM2 mask decoder on point / box prompts.

        Returns:
            masks:           ``(C, H, W)`` bool array at original resolution.
            iou_predictions: ``(C,)`` float array.
            low_res_masks:   ``(1, C, 256, 256)`` float tensor of logits, kept
                             on device so it can be fed straight into
                             :meth:`predict_alpha`.
        """
        if not self._is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before prediction.")

        mask_input_t, unnorm_coords, labels, unnorm_box = self._prep_prompts(
            point_coords, point_labels, box, mask_input, normalize_coords
        )
        masks, iou_predictions, low_res_masks = self._predict(
            unnorm_coords,
            labels,
            unnorm_box,
            mask_input_t,
            multimask_output,
            return_logits=False,
        )
        masks_np = masks.squeeze(0).float().detach().cpu().numpy() > 0
        ious_np = iou_predictions.squeeze(0).float().detach().cpu().numpy()
        return masks_np, ious_np, low_res_masks

    # ------------------------------------------------------------------
    # Stage 2 — coarse mask -> alpha matte
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_alpha(
        self,
        mask_input: Union[np.ndarray, torch.Tensor],
        output_hw: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Regress a soft alpha matte from a coarse binary mask.

        Args:
            mask_input: Coarse foreground mask. Accepts either
                * a ``(H, W)`` float / bool array in image space (any size), or
                * a ``(1, 1, 256, 256)`` tensor of SAM low-res logits / values.
                Anything ``> 0`` is treated as foreground.
            output_hw: Target ``(H, W)`` of the returned alpha. Defaults to the
                original image size passed to :meth:`set_image`.

        Returns:
            ``(H, W)`` float32 alpha in ``[0, 1]``.
        """
        if not self._is_image_set or self._input_image is None:
            raise RuntimeError("An image must be set with .set_image(...) before prediction.")

        if output_hw is None:
            output_hw = tuple(self._orig_hw[0])

        mask_256 = self._to_mask_input_256(mask_input)
        mask_inputs = (mask_256 > 0.0).float().to(self.device)

        alpha, _alpha_pyramid, _unknown = self.model._forward_alpha_heads(
            mask_inputs=mask_inputs,
            high_res_features=self.all_features,
            image=self._input_image,
        )

        alpha = F.interpolate(
            alpha.float(),
            size=(int(output_hw[0]), int(output_hw[1])),
            mode="bilinear",
            align_corners=False,
        )
        alpha_np = alpha.squeeze(0).squeeze(0).clamp_(0.0, 1.0).detach().cpu().numpy()
        return alpha_np.astype(np.float32)

    def _to_mask_input_256(self, mask_input: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Normalise any supported mask representation to a ``1x1x256x256`` tensor."""
        if isinstance(mask_input, np.ndarray):
            t = torch.from_numpy(np.ascontiguousarray(mask_input)).float()
        else:
            t = mask_input.float()

        if t.dim() == 2:
            t = t[None, None]
        elif t.dim() == 3:
            t = t[None]
        if t.dim() != 4:
            raise ValueError(f"mask_input must be 2D/3D/4D, got shape {tuple(t.shape)}")

        # Keep only one channel (SAM may return C>1 with multimask_output).
        if t.shape[1] > 1:
            t = t[:, :1]

        t = t.to(self.device)
        if t.shape[-2:] != (MATTING_MASK_INPUT_SIZE, MATTING_MASK_INPUT_SIZE):
            # Map {0,1} to signed logits before resizing so bilinear
            # interpolation keeps a clean 0-crossing at the boundary.
            if t.min() >= 0.0 and t.max() <= 1.0:
                t = t * 20.0 - 10.0
            t = F.interpolate(
                t,
                size=(MATTING_MASK_INPUT_SIZE, MATTING_MASK_INPUT_SIZE),
                mode="bilinear",
                align_corners=False,
            )
        return t

    # ------------------------------------------------------------------
    # Upstream-compatible entry point
    # ------------------------------------------------------------------

    def predict(
        self,
        img=None,
        raw_mask=None,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[Union[np.ndarray, torch.Tensor]] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords=True,
    ) -> Tuple[None, np.ndarray, None]:
        """Upstream SAM2Matting signature: ``(_, alpha, _) = predict(img, raw_mask, mask_input)``.

        ``img`` and ``raw_mask`` are only used for their shape / device; the
        cached values from :meth:`set_image` are used when they are omitted.
        """
        if mask_input is None:
            raise ValueError("predict() requires mask_input; use predict_binary() for prompts.")
        if img is not None:
            self._input_image = img.to(self.device)

        output_hw = None
        if raw_mask is not None:
            output_hw = tuple(raw_mask.shape[-2:])

        alpha = self.predict_alpha(mask_input, output_hw=output_hw)
        return None, alpha[None, ...], None

    # ------------------------------------------------------------------
    # Prompt handling (identical to SAM2ImagePredictor)
    # ------------------------------------------------------------------

    def _prep_prompts(
        self, point_coords, point_labels, box, mask_logits, normalize_coords, img_idx=-1
    ):
        unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = torch.as_tensor(
                point_coords, dtype=torch.float, device=self.device
            )
            unnorm_coords = self._transforms.transform_coords(
                point_coords, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )
            labels = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            if len(unnorm_coords.shape) == 2:
                unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]
        if box is not None:
            box = torch.as_tensor(box, dtype=torch.float, device=self.device)
            unnorm_box = self._transforms.transform_boxes(
                box, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )
        if mask_logits is not None:
            mask_input = torch.as_tensor(
                mask_logits, dtype=torch.float, device=self.device
            )
            if len(mask_input.shape) == 3:
                mask_input = mask_input[None, :, :, :]
        return mask_input, unnorm_coords, labels, unnorm_box

    @torch.no_grad()
    def _predict(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        img_idx: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )

        if point_coords is not None:
            concat_points = (point_coords, point_labels)
        else:
            concat_points = None

        if boxes is not None:
            box_coords = boxes.reshape(-1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
            box_labels = box_labels.repeat(boxes.size(0), 1)

            if concat_points is not None:
                concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
                concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                concat_points = (concat_coords, concat_labels)
            else:
                concat_points = (box_coords, box_labels)

        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=mask_input,
        )

        batched_mode = concat_points is not None and concat_points[0].shape[0] > 1
        high_res_features = [
            feat_level[img_idx].unsqueeze(0)
            for feat_level in self._features["high_res_feats"]
        ]
        low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
            image_embeddings=self._features["image_embed"][img_idx].unsqueeze(0),
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )

        masks = self._transforms.postprocess_masks(low_res_masks, self._orig_hw[img_idx])
        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        if not return_logits:
            masks = masks > self.mask_threshold

        return masks, iou_predictions, low_res_masks

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def get_image_embedding(self) -> torch.Tensor:
        if not self._is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) to generate an embedding.")
        assert self._features is not None, "Features must exist if an image has been set."
        return self._features["image_embed"]

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_predictor(self) -> None:
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        self._input_image = None
        self.all_features = None
