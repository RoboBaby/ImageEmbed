"""Patch-level reranking with MaxSim scoring."""

import logging
from typing import List, Literal, Optional, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from src.config import Config

logger = logging.getLogger(__name__)


def compute_maxsim(
    query_patches: np.ndarray,
    doc_patches: np.ndarray,
    aggregation: Literal["mean", "sum", "max"] = "mean",
) -> float:
    """
    Compute MaxSim score between query patches and document patches.

    MaxSim: For each query patch, find max similarity with all doc patches,
    then aggregate these maxima.

    Args:
        query_patches: Query patch embeddings [M, D] (L2-normalized)
        doc_patches: Document patch embeddings [N, D] (L2-normalized)
        aggregation: How to aggregate max similarities ("mean", "sum", or "max")

    Returns:
        MaxSim score (higher is better)
    """
    if len(query_patches) == 0 or len(doc_patches) == 0:
        return 0.0

    # Compute similarity matrix [M, N]
    # Since vectors are L2-normalized, dot product = cosine similarity
    sim_matrix = np.dot(query_patches, doc_patches.T)

    # For each query patch, find max similarity across all doc patches
    max_sims = np.max(sim_matrix, axis=1)  # [M]

    # Aggregate
    if aggregation == "mean":
        return float(np.mean(max_sims))
    elif aggregation == "sum":
        return float(np.sum(max_sims))
    elif aggregation == "max":
        return float(np.max(max_sims))
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


def rerank_with_patches(
    query_patches: np.ndarray,
    candidates: List[PointStruct],
    using_multivector: bool,
    collection: str,
    client: Optional[QdrantClient] = None,
    aggregation: Literal["mean", "sum", "max"] = "mean",
    alpha: float = 0.5,
) -> List[Tuple[PointStruct, float]]:
    """
    Rerank candidates using patch-level MaxSim.

    Combines global similarity (from initial retrieval) with patch-level MaxSim:
    final_score = alpha * global_score + (1 - alpha) * maxsim_score

    Args:
        query_patches: Query patch embeddings [M, D]
        candidates: List of candidate points from initial retrieval
        using_multivector: Whether multivector mode is enabled
        collection: Collection name (for fallback mode)
        client: Qdrant client (needed for fallback mode)
        aggregation: MaxSim aggregation method
        alpha: Weight for global score (0.0 = patches only, 1.0 = global only)

    Returns:
        List of (point, reranked_score) tuples sorted by score (descending)
    """
    if len(query_patches) == 0:
        # No query patches, use original scores
        return [(p, p.score if p.score else 0.0) for p in candidates]

    reranked = []

    for candidate in candidates:
        global_score = candidate.score if candidate.score else 0.0

        # Extract document patches
        if using_multivector:
            # Patches stored in named vector
            if hasattr(candidate, "vector") and isinstance(candidate.vector, dict):
                doc_patches_list = candidate.vector.get("patches", [])
                if doc_patches_list:
                    doc_patches = np.array(doc_patches_list, dtype=np.float32)
                else:
                    doc_patches = np.array([], dtype=np.float32).reshape(0, query_patches.shape[1])
            else:
                doc_patches = np.array([], dtype=np.float32).reshape(0, query_patches.shape[1])
        else:
            # Fallback mode: need to fetch patches from separate collection
            if client is None:
                logger.warning("Fallback mode requires client, skipping patch reranking")
                reranked.append((candidate, global_score))
                continue

            try:
                patches_collection = f"{collection}_patches"
                patch_points = client.scroll(
                    collection_name=patches_collection,
                    scroll_filter={"must": [{"key": "frame_id", "match": {"value": candidate.id}}]},
                    limit=1000,
                    with_vectors=True,
                )[0]

                if patch_points:
                    doc_patches = np.array(
                        [p.vector for p in patch_points], dtype=np.float32
                    )
                else:
                    doc_patches = np.array([], dtype=np.float32).reshape(0, query_patches.shape[1])
            except Exception as e:
                logger.warning(f"Failed to fetch patches for {candidate.id}: {e}")
                reranked.append((candidate, global_score))
                continue

        # Compute MaxSim score
        if len(doc_patches) > 0:
            maxsim_score = compute_maxsim(query_patches, doc_patches, aggregation)
        else:
            maxsim_score = 0.0

        # Combine scores
        final_score = alpha * global_score + (1 - alpha) * maxsim_score

        reranked.append((candidate, final_score))

    # Sort by final score (descending)
    reranked.sort(key=lambda x: x[1], reverse=True)

    return reranked


def rerank_with_query_image(
    query_image_patches: np.ndarray,
    query_global_embedding: np.ndarray,
    candidates: List[PointStruct],
    using_multivector: bool,
    collection: str,
    client: Optional[QdrantClient] = None,
    aggregation: Literal["mean", "sum", "max"] = "mean",
    alpha: float = 0.5,
) -> List[Tuple[PointStruct, float, float, float]]:
    """
    Rerank candidates with both global and patch-level scores.

    Args:
        query_image_patches: Query image patch embeddings [M, D]
        query_global_embedding: Query global embedding [D]
        candidates: List of candidate points
        using_multivector: Whether using multivector mode
        collection: Collection name
        client: Qdrant client (for fallback mode)
        aggregation: MaxSim aggregation
        alpha: Weight for global vs patch scores

    Returns:
        List of (point, final_score, global_score, maxsim_score) tuples
    """
    results = []

    for candidate in candidates:
        # Compute global similarity
        if hasattr(candidate, "vector"):
            if using_multivector and isinstance(candidate.vector, dict):
                doc_global = np.array(candidate.vector.get("global", []), dtype=np.float32)
            elif not using_multivector:
                doc_global = np.array(candidate.vector, dtype=np.float32)
            else:
                doc_global = np.array([], dtype=np.float32)

            if len(doc_global) > 0:
                global_score = float(np.dot(query_global_embedding, doc_global))
            else:
                global_score = candidate.score if candidate.score else 0.0
        else:
            global_score = candidate.score if candidate.score else 0.0

        # Get document patches
        if using_multivector:
            if hasattr(candidate, "vector") and isinstance(candidate.vector, dict):
                doc_patches_list = candidate.vector.get("patches", [])
                if doc_patches_list:
                    doc_patches = np.array(doc_patches_list, dtype=np.float32)
                else:
                    doc_patches = np.array([], dtype=np.float32).reshape(0, query_image_patches.shape[1] if len(query_image_patches) > 0 else 512)
            else:
                doc_patches = np.array([], dtype=np.float32).reshape(0, query_image_patches.shape[1] if len(query_image_patches) > 0 else 512)
        else:
            # Fallback mode
            if client is None:
                logger.warning("Client required for fallback mode patch reranking")
                results.append((candidate, global_score, global_score, 0.0))
                continue

            try:
                patches_collection = f"{collection}_patches"
                patch_points = client.scroll(
                    collection_name=patches_collection,
                    scroll_filter={"must": [{"key": "frame_id", "match": {"value": candidate.id}}]},
                    limit=1000,
                    with_vectors=True,
                )[0]

                if patch_points:
                    doc_patches = np.array([p.vector for p in patch_points], dtype=np.float32)
                else:
                    doc_patches = np.array([], dtype=np.float32).reshape(0, query_image_patches.shape[1] if len(query_image_patches) > 0 else 512)
            except Exception as e:
                logger.warning(f"Failed to fetch patches: {e}")
                results.append((candidate, global_score, global_score, 0.0))
                continue

        # Compute MaxSim
        if len(query_image_patches) > 0 and len(doc_patches) > 0:
            maxsim_score = compute_maxsim(query_image_patches, doc_patches, aggregation)
        else:
            maxsim_score = 0.0

        # Combine scores
        final_score = alpha * global_score + (1 - alpha) * maxsim_score

        results.append((candidate, final_score, global_score, maxsim_score))

    # Sort by final score
    results.sort(key=lambda x: x[1], reverse=True)

    return results
