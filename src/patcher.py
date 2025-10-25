"""Advanced sliding-window patch generation with overlap and context padding."""

from typing import Any, Dict, List

import numpy as np
from PIL import Image


def compute_patches(
    width: int,
    height: int,
    window: int,
    stride: int,
    context: float,
    scales: List[float],
) -> List[Dict[str, Any]]:
    """
    Generate sliding-window patches with overlap, context padding, and multi-scale.

    For each scale:
    1. Generate a grid of windows with specified stride
    2. Ensure coverage to image edges (clip final windows to bounds)
    3. Add context padding around each window
    4. Return patch specifications with both window and context coordinates

    Args:
        width: Image width in pixels
        height: Image height in pixels
        window: Base window size (will be scaled)
        stride: Base stride between windows (will be scaled)
        context: Context padding ratio (e.g., 0.15 for 15% padding)
        scales: List of scale factors to apply

    Returns:
        List of patch specifications, each containing:
            - x, y, w, h: Core window coordinates
            - scale: Scale factor used
            - x_ctx, y_ctx, w_ctx, h_ctx: Context-padded coordinates
    """
    patches = []

    for scale in scales:
        # Scale window and stride
        scaled_window = round(window * scale)
        scaled_stride = round(stride * scale)

        # Ensure minimum stride of 1
        scaled_stride = max(1, scaled_stride)

        # Generate grid positions
        x_positions = list(range(0, width, scaled_stride))
        y_positions = list(range(0, height, scaled_stride))

        # Ensure we cover the edges
        if x_positions and x_positions[-1] + scaled_window < width:
            x_positions.append(width - scaled_window)
        if y_positions and y_positions[-1] + scaled_window < height:
            y_positions.append(height - scaled_window)

        # Handle case where image is smaller than window
        if not x_positions or x_positions[0] < 0:
            x_positions = [0]
        if not y_positions or y_positions[0] < 0:
            y_positions = [0]

        for y in y_positions:
            for x in x_positions:
                # Core window coordinates (clamp to image bounds)
                x_win = max(0, min(x, width - 1))
                y_win = max(0, min(y, height - 1))
                w_win = min(scaled_window, width - x_win)
                h_win = min(scaled_window, height - y_win)

                # Calculate context padding
                max_dim = max(w_win, h_win)
                pad = round(max_dim * context)

                # Context-padded coordinates (clamp to image bounds)
                x_ctx = max(0, x_win - pad)
                y_ctx = max(0, y_win - pad)
                x_ctx_end = min(width, x_win + w_win + pad)
                y_ctx_end = min(height, y_win + h_win + pad)
                w_ctx = x_ctx_end - x_ctx
                h_ctx = y_ctx_end - y_ctx

                patches.append(
                    {
                        "x": int(x_win),
                        "y": int(y_win),
                        "w": int(w_win),
                        "h": int(h_win),
                        "scale": float(scale),
                        "x_ctx": int(x_ctx),
                        "y_ctx": int(y_ctx),
                        "w_ctx": int(w_ctx),
                        "h_ctx": int(h_ctx),
                    }
                )

    # Ensure at least one patch (use entire image if needed)
    if not patches:
        pad = round(max(width, height) * context)
        patches.append(
            {
                "x": 0,
                "y": 0,
                "w": width,
                "h": height,
                "scale": 1.0,
                "x_ctx": 0,
                "y_ctx": 0,
                "w_ctx": width,
                "h_ctx": height,
            }
        )

    return patches


def subsample_patches(patches: List[Dict[str, Any]], max_patches: int) -> List[Dict[str, Any]]:
    """
    Uniformly subsample patches to a maximum count.

    Preserves spatial coverage by sampling evenly across the patch list.

    Args:
        patches: List of patch specifications
        max_patches: Maximum number of patches to keep

    Returns:
        Subsampled list of patches
    """
    if len(patches) <= max_patches:
        return patches

    # Uniform sampling with equal spacing
    indices = np.linspace(0, len(patches) - 1, max_patches, dtype=int)
    return [patches[i] for i in indices]


def crop_patch(img: Image.Image, patch_spec: Dict[str, Any]) -> Image.Image:
    """
    Crop a patch from an image using context coordinates.

    Does NOT resize - the model's preprocess transform will handle that.

    Args:
        img: PIL Image
        patch_spec: Patch specification with x_ctx, y_ctx, w_ctx, h_ctx

    Returns:
        Cropped PIL Image
    """
    x = patch_spec["x_ctx"]
    y = patch_spec["y_ctx"]
    w = patch_spec["w_ctx"]
    h = patch_spec["h_ctx"]

    # PIL crop expects (left, upper, right, lower)
    return img.crop((x, y, x + w, y + h))
