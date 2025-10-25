"""Image I/O utilities with EXIF orientation handling."""

from io import BytesIO
from typing import Tuple

from PIL import Image, ImageOps


def open_image_safe(img_bytes: bytes) -> Image.Image:
    """
    Open an image from bytes with EXIF orientation handling.

    Automatically corrects image orientation based on EXIF data
    and converts to RGB mode.

    Args:
        img_bytes: Raw image bytes

    Returns:
        PIL Image in RGB mode with correct orientation

    Raises:
        Exception: If image cannot be opened
    """
    try:
        img = Image.open(BytesIO(img_bytes))

        # Apply EXIF orientation
        img = ImageOps.exif_transpose(img)

        # Convert to RGB (handle RGBA, grayscale, etc.)
        if img.mode != "RGB":
            img = img.convert("RGB")

        return img

    except Exception as e:
        raise Exception(f"Failed to open image: {e}")


def read_image_size(img: Image.Image) -> Tuple[int, int]:
    """
    Get image dimensions.

    Args:
        img: PIL Image

    Returns:
        Tuple of (width, height)
    """
    return img.size
