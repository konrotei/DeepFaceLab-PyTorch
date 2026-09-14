from pydantic import BaseModel
from typing import Optional


class ProjectOpenRequest(BaseModel):
    path: str


class ProjectOpenResponse(BaseModel):
    files: list[dict]
    count: int


class ImageResponse(BaseModel):
    image: str  # base64
    width: int
    height: int
    has_mask: bool


class MattingOptions(BaseModel):
    """Shared SAM2Matting settings (alpha -> binary conversion)."""

    matting_model: str = "sam2.1_tiny"  # "sam2.1_tiny" or "sam2.1_base_plus"
    alpha_threshold: float = 0.5  # alpha >= threshold -> foreground
    min_area: int = 0  # drop thresholded speckles smaller than this (px, at image res)
    fill_holes_area: int = 0  # fill background holes smaller than this (px)


class MaskPredictRequest(MattingOptions):
    image_index: int
    clicks: Optional[list[list[int]]] = None  # [[x, y, label], ...]
    box: Optional[list[int]] = None  # [x1, y1, x2, y2]
    backend: str = "sam"  # "sam" (binary, segment_anything) or "sam2matting" (alpha)


class MaskMattingRefineRequest(MattingOptions):
    """Refine an existing (binary) mask into an alpha matte with SAM2Matting."""

    image_index: int
    mask: str  # base64 encoded PNG of the current mask (any resolution)


class MaskTextRequest(BaseModel):
    image_index: int
    text: str
    backend: str = "grounded_sam2"  # "grounded_sam2" or "owl_vit"


class MaskBiSeNetRequest(BaseModel):
    image_index: int
    parts: list[str]  # ["face", "hair", ...]


class MaskSaveRequest(BaseModel):
    image_index: int
    mask: str  # base64 encoded PNG


class MaskResponse(BaseModel):
    mask: str  # base64 encoded PNG, always a hard 0/255 mask (XSeg-compatible)
    success: bool = True
    # SAM2Matting only: the raw soft alpha matte (base64 PNG, 0-255) and the
    # threshold used to derive `mask` from it, so the UI can re-threshold locally.
    alpha: Optional[str] = None
    alpha_threshold: Optional[float] = None


class UndoRequest(BaseModel):
    image_index: int


class ErrorResponse(BaseModel):
    error: str
    success: bool = False
