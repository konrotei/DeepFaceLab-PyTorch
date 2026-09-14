# MaskProcessor

MaskProcessor is an interactive mask annotation and generation tool for
DeepFaceLab XSeg training pipelines. It provides a web-based UI for
creating, editing, and exporting segmentation masks using AI backends
(SAM, Grounded SAM2, BiSeNet) and manual editing tools.

## Requirements

- **Python**: 3.10 or later
- **CUDA GPU (recommended)**: NVIDIA GPU with 8 GB+ VRAM for SAM / Grounded
  SAM2 inference. CPU fallback is supported but will be significantly slower.
- **Operating System**: Windows / Linux / macOS (Windows is primary target)

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd DeepFaceLab-Torch
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate.bat     # Windows (cmd)
venv\Scripts\Activate.ps1     # Windows (PowerShell)
```

### 3. Install PyTorch

Install PyTorch 2.x with CUDA support:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

For CPU-only:

```bash
pip install torch torchvision
```

### 4. Install MaskProcessor dependencies

```bash
cd MaskProcessor
pip install -r requirements.txt
```

### 5. Verify installation

```bash
python -m MaskProcessor
```

If the server starts successfully you will see:

```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

## Model Checkpoints

MaskProcessor requires model checkpoints for AI backends. The recommended
approach is to let the application download them automatically on first use
(the `ModelLoader` class handles this under
`MaskProcessor/core/model_loader.py`).

### Supported backends and their checkpoints

| Backend         | Checkpoint                                                    | Size   |
|-----------------|---------------------------------------------------------------|--------|
| SAM             | `sam_vit_h_4b8939.pth`                                       | 2.6 GB |
| Grounded SAM2   | `sam2_hiera_large.pt` + GroundingDINO config + checkpoint     | ~1 GB  |
| BiSeNet         | `bisenet.onnx` (bundled with xlib)                            | ~5 MB  |

### Manual download locations

Checkpoints are cached in the modelhub directory
(`<project_root>/modelhub/`) or in `~/.cache/` depending on your
configuration. You can also pre-download them:

- **SAM**: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
- **SAM2**: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

Place downloaded checkpoints in `modelhub/` at the project root.

## Running

### Start the web UI server

```bash
python -m MaskProcessor
```

Then open http://127.0.0.1:8000 in your browser.

### Configuration

The server runs on `127.0.0.1:8000` by default. To change the port:

```bash
python -m MaskProcessor --port 8080
```

## Optional: SAM2Matting (alpha matting)

[SAM2Matting](https://github.com/FudanCVL/SAM2Matting) adds a soft *alpha
matte* backend (hair strands, translucency) on top of the SAM2 backbone that
already ships in `xlib/models/sam2`. No extra Python packages are needed
beyond the SAM2 requirements (`hydra-core`, `omegaconf`, `torch>=2.x`).

Download a checkpoint from
https://huggingface.co/FudanCVL/SAM2Matting/tree/main/checkpoints and place it
next to the SAM2 checkpoints:

| UI name | File                              | Path                                                      |
|---------|-----------------------------------|-----------------------------------------------------------|
| tiny    | `SAM2Matting-SAM2.1Tiny.pt`       | `xlib/models/sam2/checkpoints/SAM2Matting-SAM2.1Tiny.pt`  |
| base+   | `SAM2Matting-SAM2.1Base+.pt`      | `xlib/models/sam2/checkpoints/SAM2Matting-SAM2.1Base+.pt` |

The SAM3-based variant is not supported (it needs the separate `sam3`
package). If the checkpoint is missing, the rest of MaskProcessor keeps
working and the SAM2Matting buttons return a `503` with the download hint.

> **Binary vs alpha.** DeepFaceLab XSeg masks are strictly binary
> (`DynamicSampleGenerator` thresholds at 0.5, `seg_ie_polys` are polygons).
> SAM2Matting returns alpha in `[0, 1]`. MaskProcessor therefore never writes
> the alpha to a DFLJPG: the server binarises it with
> `mask_ops.alpha_to_binary(alpha, threshold, min_area)` and also returns the
> raw alpha so the UI can re-threshold interactively. See the
> [Usage Guide](usage.md#sam2matting-alpha-matting).

## Next Steps

Once the server is running, head to the [Usage Guide](usage.md) for an
overview of the UI and tools.
