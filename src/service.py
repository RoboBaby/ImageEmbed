"""FastAPI service for S3 image ingestion with OpenCLIP embeddings."""

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from src.config import Config
from src.embedder import OpenCLIPEmbedder
from src.io_utils import open_image_safe, read_image_size
from src.patcher import compute_patches
from src.qdrant_store import ensure_collections, upsert_frame
from src.s3_client import fetch_image_bytes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global state
app = FastAPI(
    title="Visual Ingest Service",
    description="S3 image ingestion with OpenCLIP embeddings and Qdrant storage",
    version="0.1.0",
)

# These will be initialized on startup
qdrant_client: Optional[QdrantClient] = None
embedder: Optional[OpenCLIPEmbedder] = None
using_multivector: bool = False


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
    global qdrant_client, embedder, using_multivector

    logger.info("Starting Visual Ingest Service")
    logger.info(f"Configuration: {Config.summary()}")

    # Initialize Qdrant client
    logger.info(f"Connecting to Qdrant at {Config.QDRANT_URL}")
    qdrant_client = QdrantClient(url=Config.QDRANT_URL)

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
