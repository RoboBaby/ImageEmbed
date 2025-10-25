"""FastAPI service for S3 image ingestion and search with OpenCLIP embeddings."""

import asyncio
import io
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from PIL import Image
from qdrant_client import QdrantClient

from src.async_jobs import JobQueue, JobResult, JobStatus, get_job_queue
from src.config import Config
from src.embedder import OpenCLIPEmbedder
from src.io_utils import open_image_safe, read_image_size
from src.patcher import compute_patches
from src.qdrant_store import ensure_collections, upsert_frame
from src.s3_client import fetch_image_bytes
from src.search import (
    build_qdrant_filter,
    search_by_image,
    search_by_s3_url,
    search_by_text,
    hybrid_search,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global state
app = FastAPI(
    title="Visual Search & Ingest Service",
    description="S3 image ingestion and retrieval with OpenCLIP embeddings, patch-level reranking, and hybrid search",
    version="0.2.0",
)

# These will be initialized on startup
qdrant_client: Optional[QdrantClient] = None
embedder: Optional[OpenCLIPEmbedder] = None
using_multivector: bool = False
job_queue: Optional[JobQueue] = None


# Pydantic models
class IngestFrameRequest(BaseModel):
    """Request model for single frame ingestion."""

    s3_url: str = Field(..., description="S3 URL (s3:// or https://)")
    video_id: int = Field(..., description="Video ID")
    frame_id: int = Field(..., description="Frame ID")
    timestamp_ms: Optional[int] = Field(None, description="Optional timestamp in milliseconds")


class IngestFrameResponse(BaseModel):
    """Response model for single frame ingestion."""

    image_id: str
    video_id: int
    frame_id: int
    global_dim: int
    num_patches: int
    patch_dim: int
    durations_ms: Dict[str, float]
    stored_multivector: bool


class BatchIngestItem(BaseModel):
    """Individual item in batch ingest request."""

    s3_url: str
    video_id: int
    frame_id: int
    timestamp_ms: Optional[int] = None


class BatchIngestRequest(BaseModel):
    """Request model for batch ingestion."""

    items: List[BatchIngestItem]


class BatchIngestItemResult(BaseModel):
    """Result for a single item in batch ingestion."""

    s3_url: str
    video_id: int
    frame_id: int
    ok: bool
    image_id: Optional[str] = None
    num_patches: Optional[int] = None
    error: Optional[str] = None


class BatchIngestResponse(BaseModel):
    """Response model for batch ingestion."""

    total: int
    successful: int
    failed: int
    results: List[BatchIngestItemResult]
    total_duration_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    ok: bool
    config: Dict[str, Any]
    qdrant_reachable: bool
    model_loaded: bool
    using_multivector: bool


@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    global qdrant_client, embedder, using_multivector, job_queue

    logger.info("Starting Visual Search & Ingest Service")
    logger.info(f"Configuration: {Config.summary()}")

    # Initialize Qdrant client
    logger.info(f"Connecting to Qdrant at {Config.QDRANT_URL}")
    qdrant_client = QdrantClient(
        url=Config.QDRANT_URL,
        api_key=Config.QDRANT_API_KEY,
    )

    # Initialize embedder
    logger.info(
        f"Loading OpenCLIP model: {Config.OPENCLIP_MODEL} ({Config.OPENCLIP_PRETRAINED})"
    )
    embedder = OpenCLIPEmbedder(
        model_name=Config.OPENCLIP_MODEL,
        pretrained=Config.OPENCLIP_PRETRAINED,
    )
    logger.info(f"Model loaded on device: {embedder.device}")
    logger.info(f"Embedding dimension: {embedder.get_embedding_dim()}")

    # Ensure collections exist
    logger.info("Ensuring Qdrant collections exist")
    using_multivector = ensure_collections(
        client=qdrant_client,
        use_multivector=Config.USE_QDRANT_MULTIVECTOR,
        dim=embedder.get_embedding_dim(),
        collection=Config.QDRANT_COLLECTION,
    )
    logger.info(f"Using multivector mode: {using_multivector}")

    # Initialize job queue
    job_queue = get_job_queue()
    logger.info("Job queue initialized")

    logger.info("Service startup complete")


@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Verifies:
    - Qdrant connectivity
    - Model is loaded and can perform inference
    """
    if qdrant_client is None or embedder is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Check Qdrant connectivity
    qdrant_ok = False
    try:
        qdrant_client.get_collections()
        qdrant_ok = True
    except Exception as e:
        logger.error(f"Qdrant health check failed: {e}")

    # Check model with dummy image
    model_ok = False
    try:
        import numpy as np
        from PIL import Image

        dummy_img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        _ = embedder.embed_image(dummy_img)
        model_ok = True
    except Exception as e:
        logger.error(f"Model health check failed: {e}")

    return HealthResponse(
        ok=qdrant_ok and model_ok,
        config=Config.summary(),
        qdrant_reachable=qdrant_ok,
        model_loaded=model_ok,
        using_multivector=using_multivector,
    )


@app.post("/ingest/frame", response_model=IngestFrameResponse)
async def ingest_frame(request: IngestFrameRequest):
    """
    Ingest a single frame from S3.

    Process:
    1. Fetch image bytes from S3
    2. Open and process image
    3. Generate global embedding
    4. Generate patch embeddings
    5. Upsert to Qdrant

    Returns detailed timing information and storage confirmation.
    """
    if qdrant_client is None or embedder is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    durations = {}
    start_total = time.time()

    try:
        # Fetch image from S3
        t0 = time.time()
        try:
            img_bytes = fetch_image_bytes(request.s3_url)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"S3 fetch failed: {e}")
        durations["fetch"] = (time.time() - t0) * 1000

        # Open image
        t0 = time.time()
        img = open_image_safe(img_bytes)
        width, height = read_image_size(img)
        durations["image_open"] = (time.time() - t0) * 1000

        # Generate global embedding
        t0 = time.time()
        global_embedding = embedder.embed_image(img)
        durations["embed_global"] = (time.time() - t0) * 1000

        # Compute patch plan
        t0 = time.time()
        patch_specs = compute_patches(
            width=width,
            height=height,
            window=Config.PATCH_WINDOW,
            stride=Config.PATCH_STRIDE,
            context=Config.PATCH_CONTEXT_PAD,
            scales=Config.get_patch_scales(),
        )
        durations["patch_plan"] = (time.time() - t0) * 1000

        # Generate patch embeddings
        t0 = time.time()
        patch_embeddings, patch_metadata = embedder.embed_patches(img, patch_specs)
        durations["embed_patches"] = (time.time() - t0) * 1000

        # Upsert to Qdrant
        t0 = time.time()
        image_id = upsert_frame(
            client=qdrant_client,
            collection=Config.QDRANT_COLLECTION,
            video_id=request.video_id,
            frame_id=request.frame_id,
            s3_url=request.s3_url,
            width=width,
            height=height,
            global_embedding=global_embedding,
            patch_embeddings=patch_embeddings,
            patch_metadata=patch_metadata,
            use_multivector=using_multivector,
            timestamp_ms=request.timestamp_ms,
        )
        durations["qdrant_upsert"] = (time.time() - t0) * 1000
        durations["total"] = (time.time() - start_total) * 1000

        logger.info(
            f"Ingested frame {image_id}: {len(patch_embeddings)} patches, "
            f"{durations['total']:.1f}ms total"
        )

        return IngestFrameResponse(
            image_id=image_id,
            video_id=request.video_id,
            frame_id=request.frame_id,
            global_dim=embedder.get_embedding_dim(),
            num_patches=len(patch_embeddings),
            patch_dim=embedder.get_embedding_dim(),
            durations_ms=durations,
            stored_multivector=using_multivector,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion failed for {request.s3_url}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@app.post("/ingest/frames", response_model=BatchIngestResponse)
async def ingest_frames(request: BatchIngestRequest):
    """
    Ingest multiple frames in batch.

    Processes items with batching for GPU efficiency.
    Returns per-item results with success/error status.
    """
    if qdrant_client is None or embedder is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    start_total = time.time()
    results = []
    successful = 0
    failed = 0

    for item in request.items:
        try:
            # Process each item (reuse single-frame logic)
            img_bytes = fetch_image_bytes(item.s3_url)
            img = open_image_safe(img_bytes)
            width, height = read_image_size(img)

            # Generate embeddings
            global_embedding = embedder.embed_image(img)
            patch_specs = compute_patches(
                width=width,
                height=height,
                window=Config.PATCH_WINDOW,
                stride=Config.PATCH_STRIDE,
                context=Config.PATCH_CONTEXT_PAD,
                scales=Config.get_patch_scales(),
            )
            patch_embeddings, patch_metadata = embedder.embed_patches(img, patch_specs)

            # Upsert to Qdrant
            image_id = upsert_frame(
                client=qdrant_client,
                collection=Config.QDRANT_COLLECTION,
                video_id=item.video_id,
                frame_id=item.frame_id,
                s3_url=item.s3_url,
                width=width,
                height=height,
                global_embedding=global_embedding,
                patch_embeddings=patch_embeddings,
                patch_metadata=patch_metadata,
                use_multivector=using_multivector,
                timestamp_ms=item.timestamp_ms,
            )

            results.append(
                BatchIngestItemResult(
                    s3_url=item.s3_url,
                    video_id=item.video_id,
                    frame_id=item.frame_id,
                    ok=True,
                    image_id=image_id,
                    num_patches=len(patch_embeddings),
                )
            )
            successful += 1

        except Exception as e:
            logger.error(f"Failed to ingest {item.s3_url}: {e}")
            results.append(
                BatchIngestItemResult(
                    s3_url=item.s3_url,
                    video_id=item.video_id,
                    frame_id=item.frame_id,
                    ok=False,
                    error=str(e),
                )
            )
            failed += 1

    total_duration = (time.time() - start_total) * 1000

    logger.info(
        f"Batch ingestion complete: {successful} successful, {failed} failed, "
        f"{total_duration:.1f}ms total"
    )

    return BatchIngestResponse(
        total=len(request.items),
        successful=successful,
        failed=failed,
        results=results,
        total_duration_ms=total_duration,
    )


# ==================== SEARCH & RETRIEVAL ENDPOINTS ====================

class SearchResult(BaseModel):
    """Single search result."""

    image_id: str
    video_id: int
    frame_id: int
    score: float
    global_score: Optional[float] = None
    maxsim_score: Optional[float] = None
    s3_url: str
    width: int
    height: int
    timestamp_ms: Optional[int] = None


class TextSearchRequest(BaseModel):
    """Request for text-based search."""

    query: str = Field(..., description="Text query")
    top_k: int = Field(10, ge=1, le=100, description="Number of results")
    video_ids: Optional[List[int]] = Field(None, description="Filter by video IDs")
    frame_id_min: Optional[int] = Field(None, description="Minimum frame ID")
    frame_id_max: Optional[int] = Field(None, description="Maximum frame ID")
    timestamp_min_ms: Optional[int] = Field(None, description="Minimum timestamp (ms)")
    timestamp_max_ms: Optional[int] = Field(None, description="Maximum timestamp (ms)")


class ImageSearchRequest(BaseModel):
    """Request for image-based search (via S3 URL)."""

    s3_url: str = Field(..., description="S3 URL of query image")
    top_k: int = Field(10, ge=1, le=100, description="Number of results")
    rerank: bool = Field(False, description="Enable patch-level reranking")
    rerank_alpha: float = Field(0.5, ge=0.0, le=1.0, description="Global vs patch weight")
    video_ids: Optional[List[int]] = None
    frame_id_min: Optional[int] = None
    frame_id_max: Optional[int] = None
    timestamp_min_ms: Optional[int] = None
    timestamp_max_ms: Optional[int] = None


class HybridSearchRequest(BaseModel):
    """Request for hybrid text-image search."""

    text_query: Optional[str] = Field(None, description="Text query")
    image_s3_url: Optional[str] = Field(None, description="Image S3 URL")
    text_weight: float = Field(0.5, ge=0.0, le=1.0, description="Weight for text")
    image_weight: float = Field(0.5, ge=0.0, le=1.0, description="Weight for image")
    top_k: int = Field(10, ge=1, le=100, description="Number of results")
    rerank: bool = Field(False, description="Enable patch reranking")
    rerank_alpha: float = Field(0.5, ge=0.0, le=1.0, description="Global vs patch weight")
    video_ids: Optional[List[int]] = None
    frame_id_min: Optional[int] = None
    frame_id_max: Optional[int] = None


class SearchResponse(BaseModel):
    """Response for search queries."""

    results: List[SearchResult]
    total: int
    query_type: str
    rerank_enabled: bool = False
    duration_ms: float


@app.post("/search/text", response_model=SearchResponse)
async def search_text(request: TextSearchRequest):
    """
    Search by text query.

    Uses OpenCLIP text encoder to find visually similar images.
    """
    if qdrant_client is None or embedder is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    start = time.time()

    try:
        # Build filters
        filters = build_qdrant_filter(
            video_ids=request.video_ids,
            frame_id_min=request.frame_id_min,
            frame_id_max=request.frame_id_max,
            timestamp_min_ms=request.timestamp_min_ms,
            timestamp_max_ms=request.timestamp_max_ms,
        )

        # Search
        points = search_by_text(
            client=qdrant_client,
            embedder=embedder,
            collection=Config.QDRANT_COLLECTION,
            using_multivector=using_multivector,
            text_query=request.query,
            top_k=request.top_k,
            filters=filters,
        )

        # Format results
        results = []
        for point in points:
            results.append(
                SearchResult(
                    image_id=str(point.id),
                    video_id=point.payload.get("video_id"),
                    frame_id=point.payload.get("frame_id"),
                    score=point.score if point.score else 0.0,
                    s3_url=point.payload.get("s3_url", ""),
                    width=point.payload.get("width", 0),
                    height=point.payload.get("height", 0),
                    timestamp_ms=point.payload.get("timestamp_ms"),
                )
            )

        duration = (time.time() - start) * 1000
        logger.info(f"Text search completed: {len(results)} results in {duration:.1f}ms")

        return SearchResponse(
            results=results,
            total=len(results),
            query_type="text",
            duration_ms=duration,
        )

    except Exception as e:
        logger.error(f"Text search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


@app.post("/search/image", response_model=SearchResponse)
async def search_image_url(request: ImageSearchRequest):
    """
    Search by image (via S3 URL).

    Optionally applies patch-level MaxSim reranking for more accurate results.
    """
    if qdrant_client is None or embedder is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    start = time.time()

    try:
        # Build filters
        filters = build_qdrant_filter(
            video_ids=request.video_ids,
            frame_id_min=request.frame_id_min,
            frame_id_max=request.frame_id_max,
            timestamp_min_ms=request.timestamp_min_ms,
            timestamp_max_ms=request.timestamp_max_ms,
        )

        # Search
        results_with_scores = search_by_s3_url(
            client=qdrant_client,
            embedder=embedder,
            collection=Config.QDRANT_COLLECTION,
            using_multivector=using_multivector,
            s3_url=request.s3_url,
            top_k=request.top_k,
            filters=filters,
            rerank=request.rerank,
            rerank_alpha=request.rerank_alpha,
        )

        # Format results
        results = []
        for point, scores in results_with_scores:
            results.append(
                SearchResult(
                    image_id=str(point.id),
                    video_id=point.payload.get("video_id"),
                    frame_id=point.payload.get("frame_id"),
                    score=scores["final"],
                    global_score=scores.get("global"),
                    maxsim_score=scores.get("maxsim"),
                    s3_url=point.payload.get("s3_url", ""),
                    width=point.payload.get("width", 0),
                    height=point.payload.get("height", 0),
                    timestamp_ms=point.payload.get("timestamp_ms"),
                )
            )

        duration = (time.time() - start) * 1000
        logger.info(
            f"Image search completed: {len(results)} results, "
            f"rerank={request.rerank}, {duration:.1f}ms"
        )

        return SearchResponse(
            results=results,
            total=len(results),
            query_type="image",
            rerank_enabled=request.rerank,
            duration_ms=duration,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Image search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


@app.post("/search/image/upload", response_model=SearchResponse)
async def search_image_upload(
    file: UploadFile = File(...),
    top_k: int = 10,
    rerank: bool = False,
    rerank_alpha: float = 0.5,
):
    """
    Search by uploaded image file.

    Upload an image file directly for search (alternative to S3 URL).
    """
    if qdrant_client is None or embedder is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    start = time.time()

    try:
        # Read and open image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Search
        results_with_scores = search_by_image(
            client=qdrant_client,
            embedder=embedder,
            collection=Config.QDRANT_COLLECTION,
            using_multivector=using_multivector,
            image=image,
            top_k=top_k,
            filters=None,
            rerank=rerank,
            rerank_alpha=rerank_alpha,
        )

        # Format results
        results = []
        for point, scores in results_with_scores:
            results.append(
                SearchResult(
                    image_id=str(point.id),
                    video_id=point.payload.get("video_id"),
                    frame_id=point.payload.get("frame_id"),
                    score=scores["final"],
                    global_score=scores.get("global"),
                    maxsim_score=scores.get("maxsim"),
                    s3_url=point.payload.get("s3_url", ""),
                    width=point.payload.get("width", 0),
                    height=point.payload.get("height", 0),
                    timestamp_ms=point.payload.get("timestamp_ms"),
                )
            )

        duration = (time.time() - start) * 1000
        logger.info(f"Image upload search completed: {len(results)} results in {duration:.1f}ms")

        return SearchResponse(
            results=results,
            total=len(results),
            query_type="image_upload",
            rerank_enabled=rerank,
            duration_ms=duration,
        )

    except Exception as e:
        logger.error(f"Image upload search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


@app.post("/search/hybrid", response_model=SearchResponse)
async def search_hybrid(request: HybridSearchRequest):
    """
    Hybrid text-image search with score fusion.

    Combines text and image queries for more nuanced retrieval.
    """
    if qdrant_client is None or embedder is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    if not request.text_query and not request.image_s3_url:
        raise HTTPException(
            status_code=400,
            detail="Must provide at least one of: text_query, image_s3_url",
        )

    start = time.time()

    try:
        # Build filters
        filters = build_qdrant_filter(
            video_ids=request.video_ids,
            frame_id_min=request.frame_id_min,
            frame_id_max=request.frame_id_max,
        )

        # Hybrid search
        results_with_scores = hybrid_search(
            client=qdrant_client,
            embedder=embedder,
            collection=Config.QDRANT_COLLECTION,
            using_multivector=using_multivector,
            text_query=request.text_query,
            s3_url=request.image_s3_url,
            text_weight=request.text_weight,
            image_weight=request.image_weight,
            top_k=request.top_k,
            filters=filters,
            rerank=request.rerank,
            rerank_alpha=request.rerank_alpha,
        )

        # Format results
        results = []
        for point, scores in results_with_scores:
            results.append(
                SearchResult(
                    image_id=str(point.id),
                    video_id=point.payload.get("video_id"),
                    frame_id=point.payload.get("frame_id"),
                    score=scores.get("fused", 0.0),
                    global_score=scores.get("text", 0.0),
                    maxsim_score=scores.get("image", 0.0),
                    s3_url=point.payload.get("s3_url", ""),
                    width=point.payload.get("width", 0),
                    height=point.payload.get("height", 0),
                    timestamp_ms=point.payload.get("timestamp_ms"),
                )
            )

        duration = (time.time() - start) * 1000
        logger.info(f"Hybrid search completed: {len(results)} results in {duration:.1f}ms")

        return SearchResponse(
            results=results,
            total=len(results),
            query_type="hybrid",
            rerank_enabled=request.rerank,
            duration_ms=duration,
        )

    except Exception as e:
        logger.error(f"Hybrid search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


# ==================== ASYNC BATCH INGESTION ====================

class AsyncBatchIngestRequest(BaseModel):
    """Request for async batch ingestion."""

    items: List[BatchIngestItem]


class AsyncBatchIngestResponse(BaseModel):
    """Response for async batch ingestion."""

    job_id: str
    status: str
    total_items: int
    message: str


class JobStatusResponse(BaseModel):
    """Response for job status query."""

    job_id: str
    status: str
    total_items: int
    processed_items: int
    successful_items: int
    failed_items: int
    results: List[BatchIngestItemResult]
    error: Optional[str] = None
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    duration_seconds: Optional[float] = None


async def process_ingest_item(item: BatchIngestItem) -> JobResult:
    """Process a single ingestion item (for async jobs)."""
    try:
        # Fetch and process
        img_bytes = fetch_image_bytes(item.s3_url)
        img = open_image_safe(img_bytes)
        width, height = read_image_size(img)

        # Generate embeddings
        global_embedding = embedder.embed_image(img)
        patch_specs = compute_patches(
            width=width,
            height=height,
            window=Config.PATCH_WINDOW,
            stride=Config.PATCH_STRIDE,
            context=Config.PATCH_CONTEXT_PAD,
            scales=Config.get_patch_scales(),
        )
        patch_embeddings, patch_metadata = embedder.embed_patches(img, patch_specs)

        # Upsert
        image_id = upsert_frame(
            client=qdrant_client,
            collection=Config.QDRANT_COLLECTION,
            video_id=item.video_id,
            frame_id=item.frame_id,
            s3_url=item.s3_url,
            width=width,
            height=height,
            global_embedding=global_embedding,
            patch_embeddings=patch_embeddings,
            patch_metadata=patch_metadata,
            use_multivector=using_multivector,
            timestamp_ms=item.timestamp_ms,
        )

        return JobResult(
            s3_url=item.s3_url,
            video_id=item.video_id,
            frame_id=item.frame_id,
            ok=True,
            image_id=image_id,
            num_patches=len(patch_embeddings),
        )

    except Exception as e:
        return JobResult(
            s3_url=item.s3_url,
            video_id=item.video_id,
            frame_id=item.frame_id,
            ok=False,
            error=str(e),
        )


@app.post("/ingest/frames/async", response_model=AsyncBatchIngestResponse)
async def ingest_frames_async(request: AsyncBatchIngestRequest):
    """
    Ingest frames asynchronously in the background.

    Returns immediately with a job_id. Use GET /jobs/{job_id} to check status.
    """
    if job_queue is None:
        raise HTTPException(status_code=503, detail="Job queue not initialized")

    if not request.items:
        raise HTTPException(status_code=400, detail="No items provided")

    # Create job
    job_id = job_queue.create_job(total_items=len(request.items))

    # Start background task
    job_queue.start_job(
        job_id=job_id,
        process_func=process_ingest_item,
        items=request.items,
    )

    logger.info(f"Started async ingestion job {job_id} with {len(request.items)} items")

    return AsyncBatchIngestResponse(
        job_id=job_id,
        status="pending",
        total_items=len(request.items),
        message=f"Job started. Use GET /jobs/{job_id} to check status.",
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get status of an async ingestion job.
    """
    if job_queue is None:
        raise HTTPException(status_code=503, detail="Job queue not initialized")

    job = job_queue.get_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Convert results
    results = []
    for r in job.results:
        results.append(
            BatchIngestItemResult(
                s3_url=r.s3_url,
                video_id=r.video_id,
                frame_id=r.frame_id,
                ok=r.ok,
                image_id=r.image_id,
                num_patches=r.num_patches,
                error=r.error,
            )
        )

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status.value,
        total_items=job.total_items,
        processed_items=job.processed_items,
        successful_items=job.successful_items,
        failed_items=job.failed_items,
        results=results,
        error=job.error,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        duration_seconds=(
            job.completed_at - job.started_at
            if job.completed_at and job.started_at
            else None
        ),
    )


@app.get("/jobs")
async def list_jobs(limit: int = 50, status: Optional[str] = None):
    """
    List recent jobs with optional status filter.
    """
    if job_queue is None:
        raise HTTPException(status_code=503, detail="Job queue not initialized")

    status_filter = None
    if status:
        try:
            status_filter = JobStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Valid values: {[s.value for s in JobStatus]}",
            )

    jobs = job_queue.list_jobs(limit=limit, status=status_filter)

    return {
        "jobs": [
            {
                "job_id": j.job_id,
                "status": j.status.value,
                "total_items": j.total_items,
                "processed_items": j.processed_items,
                "successful_items": j.successful_items,
                "failed_items": j.failed_items,
                "created_at": j.created_at,
                "completed_at": j.completed_at,
            }
            for j in jobs
        ],
        "total": len(jobs),
    }
