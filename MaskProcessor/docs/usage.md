# MaskProcessor Usage Guide

## Opening a Workspace

When you start the server (`python -m MaskProcessor`) and open
http://127.0.0.1:8000, you will see the workspace browser. Click **Open
Workspace** and select a DeepFaceLab workspace directory (a folder
containing `data_dst/` and optionally `data_src/` subdirectories). The
application will scan for aligned faces and display them in the image
gallery.

To switch to a different workspace, click the folder icon in the toolbar.

## UI Layout

```
+--------------------------------------------------------------+
|  Toolbar  |  [Point] [Box] [Brush] [Pen] [Draw|Exclude] ...  |
+--------------------------------------------------------------+
|           |                                                   |
|  Gallery  |               Canvas                             |
|  (faces)  |         (image + mask layers)                     |
|           |                                                   |
|           |                                                   |
+-----------+---------------------------------------------------+
|  Right Panel                                                  |
|  - Layer list (image, mask, overlay)                          |
|  - AI backend selector                                        |
|  - Text prompt (Grounded SAM2)                                |
|  - Mask opacity slider                                        |
|  - Export buttons                                             |
+--------------------------------------------------------------+
```

### Toolbar (top)

The toolbar contains drawing tools, mode toggles, and action buttons.

### Gallery (left sidebar)

Thumbnails of detected faces in the current workspace. Click a thumbnail
to load that face into the canvas.

### Canvas (center)

The main editing area. Displays the source image with overlaid mask and
annotation layers. Mouse and keyboard interactions are forwarded to the
active tool.

### Right Panel

Contains layer controls, AI backend configuration, and export options.

## Tools

### Point Tool

Click on the image to place a point prompt. A green circle marks a
foreground (include) point; a red circle marks a background (exclude)
point. After placing one or more points, click **Run SAM** (or press
**Enter**) to generate a mask from the point prompts.

**Multi-point mode (Ctrl)**: Hold **Ctrl** while placing points to
accumulate multiple prompts before running inference. This is useful for
refining a mask by adding both foreground and background points.

### Box Tool

Click and drag on the canvas to draw a bounding box. The box defines a
region of interest for the segmentation model. Release the mouse to run
SAM inference automatically on the box region.

### Brush Tool

Paint directly on the mask with a soft circular brush. Use a foreground
brush (default) to add to the mask or a background brush to erase from
it. Adjust brush size with the slider in the right panel or with the `[`
and `]` keys.

- **Left-click**: Paint foreground (add to mask)
- **Right-click**: Paint background (remove from mask)

### Pen Tool

Click to place vertices of a polygon. The polygon is filled to create a
mask region. Double-click or click on the first vertex to close the
polygon.

- **Drag a vertex**: Move it.
- **Right-click a vertex**: Delete it.

## Draw / Exclude Mode Toggle

Located in the toolbar. When **Draw** mode is active, new annotations add
regions to the current mask. When **Exclude** mode is active, new
annotations subtract regions. This affects all tools (Point, Box, Brush,
Pen) and is independent of the AI backends.

## AI Backends

Select the active backend from the dropdown in the right panel.

### SAM (Segment Anything)

The default backend. Uses Meta's SAM model (ViT-H) for point and box
prompt segmentation. No text prompt is needed.

| Prompt type | How to use                                    |
|-------------|-----------------------------------------------|
| Point       | Click on the image, then press Enter or click Run SAM |
| Box         | Click and drag a bounding box                 |

### Grounded SAM2

Combines GroundingDINO (text-to-box) with SAM2 (box-to-mask). Enter a
text description of the object you want to segment (e.g. "face",
"nose", "glasses") and click **Run**. Grounded SAM2 will detect objects
matching the description and generate masks.

**Prompt types**: Text only. Point and box prompts are not available when
this backend is selected.

### SAM2Matting (alpha matting)

SAM2Matting produces a **soft alpha matte** instead of a hard mask — it keeps
hair strands, wisps and semi-transparent edges. Because XSeg only understands
binary masks, the alpha is never stored directly; you pick where to cut it.

**Enable it**: in the **SAM** panel switch *Backend* to **SAM2Matting**. The
Point tool (single click, Ctrl-group, box) then calls SAM2Matting instead of
SAM. The panel badge changes from `binary` to `matting`.

**SAM2MATTING panel**

| Control                     | What it does                                                                                     |
|-----------------------------|--------------------------------------------------------------------------------------------------|
| Model `tiny` / `base+`      | Which checkpoint to use (loaded lazily on first use, swapped on change).                         |
| **Threshold**               | `alpha >= threshold` becomes foreground. Re-binarises the stored alpha instantly, client-side. Lower = more hair fringe, higher = solid core only. `[` / `]` nudge by 0.05. |
| Denoise                     | Server-side: drop thresholded speckles smaller than N px (alpha at hair tips often breaks into dots). |
| **Refine current mask → Alpha** | Sends the current binary mask (brush / pen / SAM / BiSeNet / XSeg) as the coarse guide and replaces it with SAM2Matting's alpha, binarised at the current threshold. This is the upstream `inference_image_sam2.py` flow and is the quickest way to recover hair from a blobby XSeg mask. |
| Alpha preview               | Draws the alpha as a graded cyan overlay; the white band marks the current threshold contour, i.e. exactly what will be committed. |
| Clear alpha                 | Forgets the alpha (the binary mask stays as-is).                                                  |

**How the alpha is merged**: the response is applied on top of the mask as it
was before the click (respecting Draw / Exclude mode). Dragging the threshold
re-applies it on that snapshot, so your earlier brush strokes are preserved.
Undo, image navigation and changing the mask resolution clear the alpha.

**What gets saved**: always the binary canvas mask (`0` / `1`), exactly like
the other backends — `xseg_mask` raster plus `seg_ie_polys` polygons.

### BiSeNet

A lightweight face-parsing model. Does not require prompts. Click **Run**
to generate a face segmentation mask covering the full face region. Best
for quick initial masks that can be refined with manual tools.

**Prompt types**: None (fully automatic).

## Keyboard Shortcuts

| Key               | Action                                  |
|-------------------|-----------------------------------------|
| `P`               | Activate Point tool                     |
| `B`               | Activate Box tool                       |
| `R`               | Activate Brush tool                     |
| `N`               | Activate Pen tool                       |
| `Enter`           | Run AI inference (current backend)      |
| `Ctrl+Z`          | Undo last action                        |
| `Ctrl+Shift+Z`    | Redo last action                        |
| `S`               | Save current mask                       |
| `D`               | Toggle Draw / Exclude mode              |
| `[`               | Decrease brush size                     |
| `]`               | Increase brush size                     |
| `+` / `=`         | Zoom in                                 |
| `-`               | Zoom out                                |
| `0`               | Reset zoom to 100%                      |
| `Space` (hold)    | Pan mode (drag to pan canvas)           |
| `Delete`          | Clear current mask                      |
| `Escape`          | Cancel current operation                |

## Saving Masks

Masks are saved as XSeg-compatible PNG files in the workspace directory.

1. Click **Save** (`S`) or use the **Save** button in the toolbar.
2. The mask is written to `data_dst/aligned/<filename>_mask.png`.
3. After saving, you can run XSeg training on the exported masks.

### Bulk export

Use the **Export All** button in the right panel to generate masks for all
faces in the gallery using the current backend configuration.
