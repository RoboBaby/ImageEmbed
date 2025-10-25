"""Configuration management for the visual ingest service."""

import os
from typing import Optional


class Config:
    """Central configuration class reading from environment variables."""

    # Qdrant settings
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "frames_v1")
    USE_QDRANT_MULTIVECTOR: bool = os.getenv("USE_QDRANT_MULTIVECTOR", "true").lower() in (
        "true",
        "1",
        "yes",
    )

    # OpenCLIP settings
    OPENCLIP_MODEL: str = os.getenv("OPENCLIP_MODEL", "ViT-B-32")
    OPENCLIP_PRETRAINED: str = os.getenv("OPENCLIP_PRETRAINED", "laion2b_s34b_b79k")
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "64"))

    # Patching settings
    PATCH_WINDOW: int = int(os.getenv("PATCH_WINDOW", "224"))
    PATCH_STRIDE: int = int(os.getenv("PATCH_STRIDE", "112"))
    PATCH_CONTEXT_PAD: float = float(os.getenv("PATCH_CONTEXT_PAD", "0.15"))
    PATCH_SCALES: str = os.getenv("PATCH_SCALES", "1.0")
    MAX_PATCHES_PER_IMAGE: Optional[int] = (
        int(os.getenv("MAX_PATCHES_PER_IMAGE")) if os.getenv("MAX_PATCHES_PER_IMAGE") else None
    )

    # S3 settings
    AWS_REGION: Optional[str] = os.getenv("AWS_REGION")
    S3_ENDPOINT_URL: Optional[str] = os.getenv("S3_ENDPOINT_URL")
    S3_TIMEOUT_SECONDS: int = int(os.getenv("S3_TIMEOUT_SECONDS", "30"))

    @classmethod
    def get_patch_scales(cls) -> list[float]:
        """Parse PATCH_SCALES from comma-separated string to list of floats."""
        return [float(s.strip()) for s in cls.PATCH_SCALES.split(",") if s.strip()]

    @classmethod
    def summary(cls) -> dict:
        """Return a summary of current configuration."""
        return {
            "qdrant_url": cls.QDRANT_URL,
            "qdrant_collection": cls.QDRANT_COLLECTION,
            "use_multivector": cls.USE_QDRANT_MULTIVECTOR,
            "openclip_model": cls.OPENCLIP_MODEL,
            "openclip_pretrained": cls.OPENCLIP_PRETRAINED,
            "batch_size": cls.BATCH_SIZE,
            "patch_window": cls.PATCH_WINDOW,
            "patch_stride": cls.PATCH_STRIDE,
            "patch_context_pad": cls.PATCH_CONTEXT_PAD,
            "patch_scales": cls.get_patch_scales(),
            "max_patches": cls.MAX_PATCHES_PER_IMAGE,
            "s3_timeout": cls.S3_TIMEOUT_SECONDS,
        }
