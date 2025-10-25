"""Search and retrieval functionality for visual embeddings."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range, PointStruct

from src.config import Config
from src.embedder import OpenCLIPEmbedder
from src.io_utils import open_image_safe
from src.patcher import compute_patches
from src.reranker import rerank_with_query_image, rerank_with_patches
from src.s3_client import fetch_image_bytes

logger = logging.getLogger(__name__)


def build_qdrant_filter(
    video_ids: Optional[List[int]] = None,
    frame_id_min: Optional[int] = None,
    frame_id_max: Optional[int] = None,
    timestamp_min_ms: Optional[int] = None,
    timestamp_max_ms: Optional[int] = None,
) -> Optional[Filter]:
    """
    Build Qdrant filter from search parameters.

    Args:
        video_ids: List of video IDs to filter by
        frame_id_min: Minimum frame ID (inclusive)
        frame_id_max: Maximum frame ID (inclusive)
        timestamp_min_ms: Minimum timestamp in milliseconds (inclusive)
        timestamp_max_ms: Maximum timestamp in milliseconds (inclusive)

    Returns:
        Qdrant Filter object or None if no filters specified
    """
    conditions = []

    if video_ids:
        # Match any of the video IDs
        for vid in video_ids:
            conditions.append(FieldCondition(key="video_id", match=MatchValue(value=vid)))

    if frame_id_min is not None or frame_id_max is not None:
        conditions.append(
            FieldCondition(
                key="frame_id",
                range=Range(
                    gte=frame_id_min,
                    lte=frame_id_max,
                ),
            )
        )

    if timestamp_min_ms is not None or timestamp_max_ms is not None:
        conditions.append(
            FieldCondition(
                key="timestamp_ms",
                range=Range(
                    gte=timestamp_min_ms,
                    lte=timestamp_max_ms,
                ),
            )
        )

    if not conditions:
        return None

    # If multiple video_ids, use "should" (OR), otherwise use "must" (AND)
    if video_ids and len(video_ids) > 1 and len(conditions) == len(video_ids):
        return Filter(should=conditions)
    else:
        return Filter(must=conditions)


def search_by_text(
    client: QdrantClient,
    embedder: OpenCLIPEmbedder,
    collection: str,
    using_multivector: bool,
    text_query: str,
    top_k: int = 10,
    filters: Optional[Filter] = None,
) -> List[PointStruct]:
    """
    Search by text query.

    Args:
        client: Qdrant client
        embedder: OpenCLIP embedder
        collection: Collection name
        using_multivector: Whether multivector mode is enabled
        text_query: Text search query
        top_k: Number of results to return
        filters: Optional Qdrant filters

    Returns:
        List of matching points with scores
    """
    # Encode text query
    text_embedding = embedder.embed_text(text_query)

    # Search using global embeddings
    if using_multivector:
        search_results = client.search(
            collection_name=collection,
            query_vector=("global", text_embedding.tolist()),
            limit=top_k,
            query_filter=filters,
            with_payload=True,
            with_vectors=False,
        )
    else:
        search_results = client.search(
            collection_name=collection,
            query_vector=text_embedding.tolist(),
            limit=top_k,
            query_filter=filters,
            with_payload=True,
            with_vectors=False,
        )

    return search_results


def search_by_image(
    client: QdrantClient,
    embedder: OpenCLIPEmbedder,
    collection: str,
    using_multivector: bool,
    image: Image.Image,
    top_k: int = 10,
    filters: Optional[Filter] = None,
    rerank: bool = False,
    rerank_top_n: Optional[int] = None,
    rerank_alpha: float = 0.5,
) -> List[Tuple[PointStruct, Dict[str, float]]]:
    """
    Search by image query with optional patch-level reranking.

    Args:
        client: Qdrant client
        embedder: OpenCLIP embedder
        collection: Collection name
        using_multivector: Whether multivector mode is enabled
        image: Query image
        top_k: Number of final results
        filters: Optional Qdrant filters
        rerank: Whether to apply patch-level reranking
        rerank_top_n: Number of candidates to fetch for reranking (default: top_k * 3)
        rerank_alpha: Weight for global vs patch scores (0.0-1.0)

    Returns:
        List of (point, scores_dict) tuples where scores_dict contains:
        - "final": Final score (after reranking if enabled)
        - "global": Global similarity score
        - "maxsim": MaxSim score (if reranking enabled)
    """
    # Generate global embedding
    global_embedding = embedder.embed_image(image)

    # Determine how many candidates to fetch
    fetch_limit = rerank_top_n if rerank and rerank_top_n else (top_k * 3 if rerank else top_k)

    # Search using global embeddings
    if using_multivector:
        candidates = client.search(
            collection_name=collection,
            query_vector=("global", global_embedding.tolist()),
            limit=fetch_limit,
            query_filter=filters,
            with_payload=True,
            with_vectors=rerank,  # Fetch vectors if reranking
        )
    else:
        candidates = client.search(
            collection_name=collection,
            query_vector=global_embedding.tolist(),
            limit=fetch_limit,
            query_filter=filters,
            with_payload=True,
            with_vectors=rerank,
        )

    # Apply reranking if requested
    if rerank:
        logger.info(f"Reranking {len(candidates)} candidates with patch-level MaxSim")

        # Generate query patches
        width, height = image.size
        patch_specs = compute_patches(
            width=width,
            height=height,
            window=Config.PATCH_WINDOW,
            stride=Config.PATCH_STRIDE,
            context=Config.PATCH_CONTEXT_PAD,
            scales=Config.get_patch_scales(),
        )
        query_patches, _ = embedder.embed_patches(image, patch_specs)

        # Rerank
        reranked = rerank_with_query_image(
            query_image_patches=query_patches,
            query_global_embedding=global_embedding,
            candidates=candidates,
            using_multivector=using_multivector,
            collection=collection,
            client=client,
            alpha=rerank_alpha,
        )

        # Format results
        results = []
        for point, final_score, global_score, maxsim_score in reranked[:top_k]:
            scores = {
                "final": final_score,
                "global": global_score,
                "maxsim": maxsim_score,
            }
            results.append((point, scores))

        return results
    else:
        # No reranking, just use global scores
        results = []
        for point in candidates[:top_k]:
            scores = {
                "final": point.score if point.score else 0.0,
                "global": point.score if point.score else 0.0,
            }
            results.append((point, scores))

        return results


def search_by_s3_url(
    client: QdrantClient,
    embedder: OpenCLIPEmbedder,
    collection: str,
    using_multivector: bool,
    s3_url: str,
    top_k: int = 10,
    filters: Optional[Filter] = None,
    rerank: bool = False,
    rerank_top_n: Optional[int] = None,
    rerank_alpha: float = 0.5,
) -> List[Tuple[PointStruct, Dict[str, float]]]:
    """
    Search by S3 image URL.

    Args:
        client: Qdrant client
        embedder: OpenCLIP embedder
        collection: Collection name
        using_multivector: Whether multivector mode is enabled
        s3_url: S3 URL of query image
        top_k: Number of results
        filters: Optional filters
        rerank: Whether to rerank with patches
        rerank_top_n: Number of candidates for reranking
        rerank_alpha: Global vs patch weight

    Returns:
        List of (point, scores_dict) tuples
    """
    # Fetch and open image
    img_bytes = fetch_image_bytes(s3_url)
    image = open_image_safe(img_bytes)

    # Search by image
    return search_by_image(
        client=client,
        embedder=embedder,
        collection=collection,
        using_multivector=using_multivector,
        image=image,
        top_k=top_k,
        filters=filters,
        rerank=rerank,
        rerank_top_n=rerank_top_n,
        rerank_alpha=rerank_alpha,
    )


def hybrid_search(
    client: QdrantClient,
    embedder: OpenCLIPEmbedder,
    collection: str,
    using_multivector: bool,
    text_query: Optional[str] = None,
    image_query: Optional[Image.Image] = None,
    s3_url: Optional[str] = None,
    text_weight: float = 0.5,
    image_weight: float = 0.5,
    top_k: int = 10,
    filters: Optional[Filter] = None,
    rerank: bool = False,
    rerank_top_n: Optional[int] = None,
    rerank_alpha: float = 0.5,
) -> List[Tuple[PointStruct, Dict[str, float]]]:
    """
    Hybrid text-image search with score fusion.

    Performs both text and image searches, then fuses results using weighted scores.

    Args:
        client: Qdrant client
        embedder: OpenCLIP embedder
        collection: Collection name
        using_multivector: Whether multivector mode is enabled
        text_query: Optional text query
        image_query: Optional image query
        s3_url: Optional S3 URL (alternative to image_query)
        text_weight: Weight for text similarity (0.0-1.0)
        image_weight: Weight for image similarity (0.0-1.0)
        top_k: Number of final results
        filters: Optional filters
        rerank: Whether to rerank with patches
        rerank_top_n: Number of candidates for reranking
        rerank_alpha: Global vs patch weight for reranking

    Returns:
        List of (point, scores_dict) tuples with fused scores
    """
    if not text_query and not image_query and not s3_url:
        raise ValueError("Must provide at least one of: text_query, image_query, s3_url")

    # Load image if S3 URL provided
    if s3_url and not image_query:
        img_bytes = fetch_image_bytes(s3_url)
        image_query = open_image_safe(img_bytes)

    # Perform searches
    text_results = {}
    image_results = {}

    if text_query:
        text_points = search_by_text(
            client=client,
            embedder=embedder,
            collection=collection,
            using_multivector=using_multivector,
            text_query=text_query,
            top_k=top_k * 2,  # Fetch more for fusion
            filters=filters,
        )
        for point in text_points:
            text_results[point.id] = point.score if point.score else 0.0

    if image_query:
        image_points_with_scores = search_by_image(
            client=client,
            embedder=embedder,
            collection=collection,
            using_multivector=using_multivector,
            image=image_query,
            top_k=top_k * 2,
            filters=filters,
            rerank=rerank,
            rerank_top_n=rerank_top_n,
            rerank_alpha=rerank_alpha,
        )
        for point, scores in image_points_with_scores:
            image_results[point.id] = scores["final"]

    # Fuse scores
    all_ids = set(text_results.keys()) | set(image_results.keys())
    fused_scores = {}

    for point_id in all_ids:
        text_score = text_results.get(point_id, 0.0)
        image_score = image_results.get(point_id, 0.0)
        fused_score = text_weight * text_score + image_weight * image_score
        fused_scores[point_id] = {
            "fused": fused_score,
            "text": text_score,
            "image": image_score,
        }

    # Sort by fused score
    sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x]["fused"], reverse=True)

    # Fetch full points for top results
    top_ids = sorted_ids[:top_k]
    points = client.retrieve(
        collection_name=collection,
        ids=top_ids,
        with_payload=True,
        with_vectors=False,
    )

    # Build results preserving order
    id_to_point = {p.id: p for p in points}
    results = []
    for point_id in top_ids:
        if point_id in id_to_point:
            results.append((id_to_point[point_id], fused_scores[point_id]))

    return results
