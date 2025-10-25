"""Qdrant storage with multivector support for image and patch embeddings."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PointStruct,
    VectorParams,
)

# Try to import MultiVectorConfig (available in newer qdrant-client versions)
try:
    from qdrant_client.models import MultiVectorConfig

    MULTIVECTOR_AVAILABLE = True
except ImportError:
    MULTIVECTOR_AVAILABLE = False

logger = logging.getLogger(__name__)


def ensure_collections(
    client: QdrantClient,
    use_multivector: bool,
    dim: int,
    collection: str,
) -> bool:
    """
    Ensure Qdrant collections exist with proper configuration.

    If use_multivector is True and MultiVectorConfig is available:
    - Create single collection with named vectors: "global" (indexed) and "patches" (multivector, not indexed)

    Otherwise (fallback):
    - Create two collections: frames for global embeddings, frame_patches for patch embeddings

    Args:
        client: Qdrant client
        use_multivector: Whether to use multivector mode
        dim: Embedding dimensionality
        collection: Base collection name

    Returns:
        True if using multivector mode, False if using fallback mode

    Raises:
        Exception: If collection creation fails
    """
    actual_multivector = use_multivector and MULTIVECTOR_AVAILABLE

    if actual_multivector:
        logger.info(f"Creating collection '{collection}' with multivector support")
        _ensure_multivector_collection(client, collection, dim)
    else:
        if use_multivector and not MULTIVECTOR_AVAILABLE:
            logger.warning(
                "MultiVectorConfig not available in qdrant-client, using fallback mode"
            )
        logger.info(f"Creating collections in fallback mode: '{collection}' and '{collection}_patches'")
        _ensure_fallback_collections(client, collection, dim)

    return actual_multivector


def _ensure_multivector_collection(client: QdrantClient, collection: str, dim: int) -> None:
    """Create collection with multivector support."""
    if client.collection_exists(collection):
        logger.info(f"Collection '{collection}' already exists")
        return

    # Create collection with named vectors
    client.create_collection(
        collection_name=collection,
        vectors_config={
            "global": VectorParams(size=dim, distance=Distance.COSINE),
            "patches": MultiVectorConfig(
                size=dim,
                distance=Distance.COSINE,
            ),
        },
        hnsw_config=HnswConfigDiff(
            m=32,
            ef_construct=128,
        ),
    )

    logger.info(f"Created multivector collection '{collection}' with dim={dim}")


def _ensure_fallback_collections(client: QdrantClient, collection: str, dim: int) -> None:
    """Create separate collections for global and patch embeddings."""
    # Create main collection for global embeddings
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(
                m=32,
                ef_construct=128,
            ),
        )
        logger.info(f"Created global collection '{collection}' with dim={dim}")
    else:
        logger.info(f"Collection '{collection}' already exists")

    # Create patches collection
    patches_collection = f"{collection}_patches"
    if not client.collection_exists(patches_collection):
        client.create_collection(
            collection_name=patches_collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(
                m=16,  # Smaller m for patches
                ef_construct=64,
            ),
        )
        logger.info(f"Created patches collection '{patches_collection}' with dim={dim}")
    else:
        logger.info(f"Collection '{patches_collection}' already exists")


def upsert_frame(
    client: QdrantClient,
    collection: str,
    video_id: int,
    frame_id: int,
    s3_url: str,
    width: int,
    height: int,
    global_embedding: np.ndarray,
    patch_embeddings: np.ndarray,
    patch_metadata: Dict[str, List],
    use_multivector: bool,
    timestamp_ms: Optional[int] = None,
) -> str:
    """
    Upsert a frame with global and patch embeddings into Qdrant.

    Args:
        client: Qdrant client
        collection: Collection name
        video_id: Video ID
        frame_id: Frame ID
        s3_url: S3 URL of the image
        width: Image width
        height: Image height
        global_embedding: Global embedding vector [D]
        patch_embeddings: Patch embeddings [M, D]
        patch_metadata: Dict with aligned arrays (patch_x, patch_y, patch_w, patch_h, patch_scale)
        use_multivector: Whether using multivector mode
        timestamp_ms: Optional timestamp in milliseconds

    Returns:
        image_id: Unique ID for this frame

    Raises:
        ValueError: If vectors contain NaN/Inf or dimensions mismatch
    """
    # Build unique image_id
    image_id = f"{video_id}:{frame_id}"

    # Validate embeddings
    _validate_embedding(global_embedding, "global")
    _validate_embedding(patch_embeddings, "patches")

    # Validate metadata alignment
    num_patches = len(patch_embeddings)
    for key, values in patch_metadata.items():
        if len(values) != num_patches:
            raise ValueError(
                f"Metadata '{key}' length {len(values)} doesn't match {num_patches} patches"
            )

    # Build base payload
    payload = {
        "image_id": image_id,
        "video_id": int(video_id),
        "frame_id": int(frame_id),
        "width": int(width),
        "height": int(height),
        "s3_url": s3_url,
    }

    if timestamp_ms is not None:
        payload["timestamp_ms"] = int(timestamp_ms)

    if use_multivector:
        _upsert_multivector(
            client,
            collection,
            image_id,
            payload,
            global_embedding,
            patch_embeddings,
            patch_metadata,
        )
    else:
        _upsert_fallback(
            client,
            collection,
            image_id,
            payload,
            global_embedding,
            patch_embeddings,
            patch_metadata,
        )

    return image_id


def _validate_embedding(embedding: np.ndarray, name: str) -> None:
    """Validate embedding array."""
    if len(embedding) > 0:
        if np.isnan(embedding).any():
            raise ValueError(f"{name} embedding contains NaN values")
        if np.isinf(embedding).any():
            raise ValueError(f"{name} embedding contains Inf values")


def _upsert_multivector(
    client: QdrantClient,
    collection: str,
    image_id: str,
    payload: Dict[str, Any],
    global_embedding: np.ndarray,
    patch_embeddings: np.ndarray,
    patch_metadata: Dict[str, List],
) -> None:
    """Upsert using multivector mode (single point with named vectors)."""
    # Add patch metadata to payload
    payload.update(
        {
            "patch_x": patch_metadata["patch_x"],
            "patch_y": patch_metadata["patch_y"],
            "patch_w": patch_metadata["patch_w"],
            "patch_h": patch_metadata["patch_h"],
            "patch_scale": patch_metadata["patch_scale"],
        }
    )

    # Convert embeddings to lists
    global_vec = global_embedding.tolist()
    patches_vec = patch_embeddings.tolist()  # List of lists: [[d1, d2, ...], ...]

    # Create point with named vectors
    point = PointStruct(
        id=image_id,
        vector={
            "global": global_vec,
            "patches": patches_vec,
        },
        payload=payload,
    )

    client.upsert(collection_name=collection, points=[point])
    logger.debug(f"Upserted frame {image_id} with {len(patch_embeddings)} patches (multivector)")


def _upsert_fallback(
    client: QdrantClient,
    collection: str,
    image_id: str,
    payload: Dict[str, Any],
    global_embedding: np.ndarray,
    patch_embeddings: np.ndarray,
    patch_metadata: Dict[str, List],
) -> None:
    """Upsert using fallback mode (separate collections for global and patches)."""
    # Upsert global embedding
    global_point = PointStruct(
        id=image_id,
        vector=global_embedding.tolist(),
        payload=payload,
    )
    client.upsert(collection_name=collection, points=[global_point])

    # Upsert patch embeddings
    if len(patch_embeddings) > 0:
        patches_collection = f"{collection}_patches"
        patch_points = []

        for i, patch_vec in enumerate(patch_embeddings):
            patch_payload = {
                "frame_id": image_id,
                "patch_index": i,
                "x": patch_metadata["patch_x"][i],
                "y": patch_metadata["patch_y"][i],
                "w": patch_metadata["patch_w"][i],
                "h": patch_metadata["patch_h"][i],
                "scale": patch_metadata["patch_scale"][i],
            }

            patch_point = PointStruct(
                id=f"{image_id}:patch:{i}",
                vector=patch_vec.tolist(),
                payload=patch_payload,
            )
            patch_points.append(patch_point)

        client.upsert(collection_name=patches_collection, points=patch_points)

    logger.debug(f"Upserted frame {image_id} with {len(patch_embeddings)} patches (fallback)")
