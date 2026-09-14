# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf


HF_MODEL_ID_TO_FILENAMES = {
    "facebook/sam2.1-hiera-tiny": (
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1_hiera_tiny.pt",
    ),
    "facebook/sam2.1-hiera-small": (
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1_hiera_small.pt",
    ),
    "facebook/sam2.1-hiera-base-plus": (
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_base_plus.pt",
    ),
    "facebook/sam2.1-hiera-large": (
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        "sam2.1_hiera_large.pt",
    ),
}


def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


# ---------------------------------------------------------------------------
# SAM2Matting (FudanCVL/SAM2Matting) — SAM2 backbone + alpha matting heads
# ---------------------------------------------------------------------------

SAM2MATTING_MODELS = {
    # key: (checkpoint file name, hydra config path relative to this package)
    "sam2.1_tiny": (
        "SAM2Matting-SAM2.1Tiny.pt",
        "configs/sam2matting/sam2matting-sam2.1tiny.yaml",
    ),
    "sam2.1_base_plus": (
        "SAM2Matting-SAM2.1Base+.pt",
        "configs/sam2matting/sam2matting-sam2.1base+.yaml",
    ),
}

# Parameter prefixes that only exist in SAM2Matting. If these are missing from
# a checkpoint the alpha heads would run with random weights, so we fail hard.
_SAM2MATTING_REQUIRED_PREFIXES = (
    "unknown_region_predictor.",
    "unknown_fusion.",
    "unknown_alpha_predictor.",
    "alpha_pred1.",
    "alpha_pred2.",
    "alpha_pred3.",
)


def build_sam2matting(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):
    """Build a ``SAM2MattingBase`` model for single-image matting.

    Mirrors :func:`build_sam2` but uses a lenient checkpoint loader: the
    upstream SAM2Matting checkpoints are saved from the video-predictor class
    and may carry a few extra / missing memory-related keys that do not
    affect image inference.
    """
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint_lenient(model, ckpt_path, required_prefixes=_SAM2MATTING_REQUIRED_PREFIXES)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def _load_checkpoint_lenient(model, ckpt_path, required_prefixes=()):
    if ckpt_path is None:
        return
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)

    critical = [k for k in missing_keys if k.startswith(required_prefixes)]
    if critical:
        logging.error("SAM2Matting checkpoint is missing alpha-head weights: %s", critical[:10])
        raise RuntimeError(
            f"Checkpoint {ckpt_path} does not contain SAM2Matting alpha heads "
            f"({len(critical)} missing keys). Download it from "
            "https://huggingface.co/FudanCVL/SAM2Matting/tree/main/checkpoints"
        )
    if missing_keys:
        logging.warning("SAM2Matting: %d non-critical missing keys (e.g. %s)", len(missing_keys), missing_keys[:3])
    if unexpected_keys:
        logging.warning("SAM2Matting: %d unexpected keys ignored (e.g. %s)", len(unexpected_keys), unexpected_keys[:3])
    logging.info("Loaded SAM2Matting checkpoint successfully")


def _hf_download(model_id):
    from huggingface_hub import hf_hub_download

    config_name, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[model_id]
    ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
    return config_name, ckpt_path


def build_sam2_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2(config_file=config_name, ckpt_path=ckpt_path, **kwargs)


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")
