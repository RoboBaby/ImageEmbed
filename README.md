# Visual Search & Ingest Service

Production-ready HTTP service for ingesting and searching images from S3 using OpenCLIP embeddings with patch-level reranking.

## Features

### Phase 2 - Full Search & Retrieval

**Ingestion:**
- Fetches images from S3 (supports both `s3://` URLs and HTTPS pre-signed URLs)
- Generates OpenCLIP embeddings:
  - **Global embedding** per image
  - **Patch embeddings** via advanced sliding-window with overlap, context padding, and optional multi-scale
- Stores vectors in Qdrant with multivector support (or fallback mode)
- Synchronous and **asynchronous batch ingestion** with job tracking

**Search & Retrieval:**
- **Text-to-image search** using OpenCLIP text encoder
- **Image-to-image search** (by S3 URL or file upload)
- **Patch-level MaxSim reranking** for improved accuracy
- **Hybrid text-image search** with configurable score fusion
- Advanced filtering (by video_id, frame_id ranges, timestamps)
- Real-time search with detailed score breakdowns

**Production Ready:**
- Comprehensive error handling and logging
- Health checks and monitoring
- Background job queue with status tracking
- Structured timing metrics

**Not included (future):**
- Object detection (GroundingDINO/SAM)
- Quantization/compression
- Distributed processing

## Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   S3/URL    │─────▶│   FastAPI    │─────▶│   Qdrant    │
│   (images)  │      │   Service    │      │  (vectors)  │
└─────────────┘      └──────────────┘      └─────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │   OpenCLIP   │
                     │  (embeddings)│
                     └──────────────┘
```

### Storage Modes

**Multivector mode (preferred):**
- Single collection with named vectors: `global` (indexed) + `patches` (multivector, not indexed)
- Patch metadata stored in point payload
- More efficient storage and retrieval

**Fallback mode:**
- Two collections: `frames_v1` for global embeddings, `frames_v1_patches` for patch embeddings
- Used when qdrant-client doesn't support `MultiVectorConfig`

## Quick Start

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- AWS credentials (for S3 access) or pre-signed URLs

### Installation

1. **Clone and install dependencies:**

```bash
cd ImageEmbed
pip install -e .
# Or for development:
pip install -e ".[dev]"
```

2. **Start Qdrant:**

```bash
docker compose up -d
```

3. **Create collections:**

```bash
python -m scripts.create_collection
```

4. **Start the service:**

```bash
uvicorn src.service:app --reload --port 8000
```

5. **Verify health:**

```bash
curl http://localhost:8000/healthz
```

## Configuration

All configuration is via environment variables with sensible defaults.

### Qdrant Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_COLLECTION` | `frames_v1` | Base collection name |
| `USE_QDRANT_MULTIVECTOR` | `true` | Use multivector mode if available |

### OpenCLIP Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENCLIP_MODEL` | `ViT-B-32` | Model architecture |
| `OPENCLIP_PRETRAINED` | `laion2b_s34b_b79k` | Pretrained weights |
| `BATCH_SIZE` | `64` | Batch size for GPU inference |

### Patching Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `PATCH_WINDOW` | `224` | Base window size (pixels) |
| `PATCH_STRIDE` | `112` | Stride between windows (50% overlap) |
| `PATCH_CONTEXT_PAD` | `0.15` | Context padding ratio (15%) |
| `PATCH_SCALES` | `1.0` | Comma-separated scales (e.g., "0.75,1.0,1.5") |
| `MAX_PATCHES_PER_IMAGE` | `None` | Max patches per image (uniform subsample if exceeded) |

### S3 Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_REGION` | (auto) | AWS region |
| `AWS_ACCESS_KEY_ID` | (boto3 default) | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | (boto3 default) | AWS secret key |
| `S3_ENDPOINT_URL` | `None` | Custom S3 endpoint (MinIO/LocalStack) |
| `S3_TIMEOUT_SECONDS` | `30` | S3 request timeout |

**Note:** The service uses standard AWS credential chain (env vars, IAM roles, etc.). For IAM role-based auth (recommended in production), no explicit credentials needed.

## API Reference

### `GET /healthz`

Health check endpoint.

**Response:**
```json
{
  "ok": true,
  "config": { ... },
  "qdrant_reachable": true,
  "model_loaded": true,
  "using_multivector": true
}
```

### `POST /ingest/frame`

Ingest a single image frame.

**Request:**
```json
{
  "s3_url": "s3://my-bucket/video123/frame456.jpg",
  "video_id": 123,
  "frame_id": 456,
  "timestamp_ms": 15342
}
```

**Response:**
```json
{
  "image_id": "123:456",
  "video_id": 123,
  "frame_id": 456,
  "global_dim": 512,
  "num_patches": 24,
  "patch_dim": 512,
  "durations_ms": {
    "fetch": 145.2,
    "image_open": 8.3,
    "embed_global": 42.1,
    "patch_plan": 1.2,
    "embed_patches": 156.8,
    "qdrant_upsert": 22.4,
    "total": 376.0
  },
  "stored_multivector": true
}
```

**Supported S3 URL formats:**
- `s3://bucket/key/path.jpg`
- `https://bucket.s3.amazonaws.com/key/path.jpg`
- `https://bucket.s3.region.amazonaws.com/key/path.jpg`
- `https://s3.region.amazonaws.com/bucket/key/path.jpg`
- Pre-signed URLs (HTTPS)

### `POST /ingest/frames`

Batch ingest multiple frames.

**Request:**
```json
{
  "items": [
    {
      "s3_url": "s3://bucket/a.jpg",
      "video_id": 1,
      "frame_id": 101
    },
    {
      "s3_url": "s3://bucket/b.jpg",
      "video_id": 1,
      "frame_id": 102
    }
  ]
}
```

**Response:**
```json
{
  "total": 2,
  "successful": 2,
  "failed": 0,
  "results": [
    {
      "s3_url": "s3://bucket/a.jpg",
      "video_id": 1,
      "frame_id": 101,
      "ok": true,
      "image_id": "1:101",
      "num_patches": 24
    },
    {
      "s3_url": "s3://bucket/b.jpg",
      "video_id": 1,
      "frame_id": 102,
      "ok": true,
      "image_id": "1:102",
      "num_patches": 32
    }
  ],
  "total_duration_ms": 842.5
}
```

### `POST /ingest/frames/async`

Asynchronous batch ingestion (non-blocking).

**Request:**
```json
{
  "items": [
    {"s3_url": "s3://bucket/a.jpg", "video_id": 1, "frame_id": 101},
    {"s3_url": "s3://bucket/b.jpg", "video_id": 1, "frame_id": 102}
  ]
}
```

**Response:**
```json
{
  "job_id": "a1b2c3d4-5678-90ab-cdef-1234567890ab",
  "status": "pending",
  "total_items": 2,
  "message": "Job started. Use GET /jobs/{job_id} to check status."
}
```

### `GET /jobs/{job_id}`

Check status of async ingestion job.

**Response:**
```json
{
  "job_id": "a1b2c3d4-...",
  "status": "completed",
  "total_items": 2,
  "processed_items": 2,
  "successful_items": 2,
  "failed_items": 0,
  "results": [...],
  "duration_seconds": 5.2
}
```

### `GET /jobs`

List recent jobs (with optional status filter).

**Query params:** `limit` (int), `status` (pending|running|completed|failed)

## Search & Retrieval Endpoints

### `POST /search/text`

Search by text query using OpenCLIP text encoder.

**Request:**
```json
{
  "query": "a person riding a bicycle",
  "top_k": 10,
  "video_ids": [1, 2],
  "frame_id_min": 100,
  "frame_id_max": 500
}
```

**Response:**
```json
{
  "results": [
    {
      "image_id": "1:234",
      "video_id": 1,
      "frame_id": 234,
      "score": 0.87,
      "s3_url": "s3://bucket/frame.jpg",
      "width": 1920,
      "height": 1080
    }
  ],
  "total": 10,
  "query_type": "text",
  "duration_ms": 42.3
}
```

### `POST /search/image`

Search by image (via S3 URL) with optional MaxSim reranking.

**Request:**
```json
{
  "s3_url": "s3://bucket/query.jpg",
  "top_k": 10,
  "rerank": true,
  "rerank_alpha": 0.5,
  "video_ids": [1]
}
```

**Response:**
```json
{
  "results": [
    {
      "image_id": "1:456",
      "video_id": 1,
      "frame_id": 456,
      "score": 0.92,
      "global_score": 0.85,
      "maxsim_score": 0.94,
      "s3_url": "s3://bucket/match.jpg",
      "width": 1920,
      "height": 1080
    }
  ],
  "total": 10,
  "query_type": "image",
  "rerank_enabled": true,
  "duration_ms": 156.7
}
```

**Reranking parameters:**
- `rerank` (bool): Enable patch-level MaxSim reranking
- `rerank_alpha` (float 0-1): Weight for global vs patch scores
  - `1.0` = global only (fast, less accurate)
  - `0.5` = balanced (default)
  - `0.0` = patches only (slow, more accurate)

### `POST /search/image/upload`

Search by uploaded image file.

**Form data:**
- `file`: Image file (multipart/form-data)
- `top_k`: Number of results (query param)
- `rerank`: Enable reranking (query param)
- `rerank_alpha`: Reranking weight (query param)

**Example:**
```bash
curl -X POST http://localhost:8000/search/image/upload \
  -F "file=@query.jpg" \
  -F "top_k=5" \
  -F "rerank=true"
```

### `POST /search/hybrid`

Hybrid text-image search with score fusion.

**Request:**
```json
{
  "text_query": "a red car",
  "image_s3_url": "s3://bucket/reference.jpg",
  "text_weight": 0.6,
  "image_weight": 0.4,
  "top_k": 10,
  "rerank": false
}
```

**Response:**
```json
{
  "results": [
    {
      "image_id": "1:789",
      "score": 0.88,
      "global_score": 0.82,
      "maxsim_score": 0.91,
      ...
    }
  ],
  "total": 10,
  "query_type": "hybrid",
  "duration_ms": 89.4
}
```

**Fusion formula:**
```
final_score = text_weight * text_score + image_weight * image_score
```

## Usage Examples

### Using curl

```bash
# Single frame ingestion
curl -X POST http://localhost:8000/ingest/frame \
  -H "Content-Type: application/json" \
  -d '{
    "s3_url": "s3://my-bucket/sample.jpg",
    "video_id": 123,
    "frame_id": 456
  }'

# Batch ingestion
curl -X POST http://localhost:8000/ingest/frames \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"s3_url": "s3://bucket/a.jpg", "video_id": 1, "frame_id": 101},
      {"s3_url": "s3://bucket/b.jpg", "video_id": 1, "frame_id": 102}
    ]
  }'
```

### Using Python

```python
import httpx

# Single frame
response = httpx.post(
    "http://localhost:8000/ingest/frame",
    json={
        "s3_url": "s3://my-bucket/sample.jpg",
        "video_id": 123,
        "frame_id": 456,
    },
    timeout=60.0,
)
print(response.json())
```

### Using pre-signed URLs

```bash
# Generate pre-signed URL (AWS CLI)
aws s3 presign s3://my-bucket/image.jpg --expires-in 3600

# Use the pre-signed URL
curl -X POST http://localhost:8000/ingest/frame \
  -H "Content-Type: application/json" \
  -d '{
    "s3_url": "https://my-bucket.s3.amazonaws.com/image.jpg?X-Amz-Algorithm=...",
    "video_id": 123,
    "frame_id": 456
  }'
```

### Search Examples

```bash
# Text search
curl -X POST http://localhost:8000/search/text \
  -H "Content-Type: application/json" \
  -d '{
    "query": "a person running",
    "top_k": 5,
    "video_ids": [1, 2]
  }'

# Image search (with reranking)
curl -X POST http://localhost:8000/search/image \
  -H "Content-Type: application/json" \
  -d '{
    "s3_url": "s3://bucket/query.jpg",
    "top_k": 10,
    "rerank": true,
    "rerank_alpha": 0.3
  }'

# Image upload search
curl -X POST http://localhost:8000/search/image/upload \
  -F "file=@query.jpg" \
  -F "top_k=5" \
  -F "rerank=true"

# Hybrid search
curl -X POST http://localhost:8000/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "text_query": "red car on highway",
    "image_s3_url": "s3://bucket/reference.jpg",
    "text_weight": 0.6,
    "image_weight": 0.4,
    "top_k": 10
  }'

# Async batch ingestion
curl -X POST http://localhost:8000/ingest/frames/async \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"s3_url": "s3://bucket/1.jpg", "video_id": 1, "frame_id": 1},
      {"s3_url": "s3://bucket/2.jpg", "video_id": 1, "frame_id": 2}
    ]
  }'

# Check job status
curl http://localhost:8000/jobs/{job_id}

# List all jobs
curl http://localhost:8000/jobs?limit=20&status=completed
```

### Python Search Examples

```python
import httpx

# Text search
response = httpx.post(
    "http://localhost:8000/search/text",
    json={
        "query": "a sunset over mountains",
        "top_k": 5,
    },
    timeout=30.0,
)
results = response.json()
for r in results["results"]:
    print(f"{r['image_id']}: score={r['score']:.3f}")

# Image search with reranking
response = httpx.post(
    "http://localhost:8000/search/image",
    json={
        "s3_url": "s3://bucket/query.jpg",
        "top_k": 10,
        "rerank": True,
        "rerank_alpha": 0.5,
    },
    timeout=60.0,
)
results = response.json()
for r in results["results"]:
    print(f"{r['image_id']}: final={r['score']:.3f}, "
          f"global={r['global_score']:.3f}, maxsim={r['maxsim_score']:.3f}")

# Async batch with polling
import time

# Start job
response = httpx.post(
    "http://localhost:8000/ingest/frames/async",
    json={"items": items},
    timeout=10.0,
)
job_id = response.json()["job_id"]

# Poll status
while True:
    response = httpx.get(f"http://localhost:8000/jobs/{job_id}")
    status = response.json()

    if status["status"] == "completed":
        print(f"Completed: {status['successful_items']}/{status['total_items']}")
        break
    elif status["status"] == "failed":
        print(f"Failed: {status['error']}")
        break

    time.sleep(1)
```

## Testing

### Run all tests

```bash
# Start Qdrant first
docker compose up -d

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html
```

### Smoke test

```bash
# Make sure service is running
uvicorn src.service:app --port 8000 &

# Run smoke test
python -m scripts.smoke_ingest \
  --s3-url s3://my-bucket/sample.jpg \
  --video-id 123 \
  --frame-id 456
```

## Advanced Configuration

### Multi-scale patching

Generate patches at multiple scales for better coverage:

```bash
export PATCH_SCALES="0.75,1.0,1.5"
```

This will create patches at 75%, 100%, and 150% of the base window size.

### Limiting patches

For very large images, limit the number of patches:

```bash
export MAX_PATCHES_PER_IMAGE=100
```

Patches will be uniformly subsampled to preserve spatial coverage.

### Custom model

Use a different OpenCLIP model:

```bash
export OPENCLIP_MODEL="ViT-L-14"
export OPENCLIP_PRETRAINED="laion2b_s32b_b82k"
```

### GPU configuration

The service auto-detects CUDA. Force CPU mode:

```python
# Modify src/embedder.py or set CUDA_VISIBLE_DEVICES=""
export CUDA_VISIBLE_DEVICES=""
```

## Production Deployment

### Docker deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e .

CMD ["uvicorn", "src.service:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment configuration

```bash
# .env file
QDRANT_URL=http://qdrant:6333
AWS_REGION=us-west-2
OPENCLIP_MODEL=ViT-B-32
BATCH_SIZE=32
```

### Monitoring

The service provides structured logs with timing information:

```
2024-01-15 10:30:45 - src.service - INFO - Ingested frame 123:456: 24 patches, 376.0ms total
```

Monitor these metrics:
- `durations_ms.total` - Total ingestion time
- `durations_ms.fetch` - S3 fetch time
- `durations_ms.embed_*` - Embedding time
- `durations_ms.qdrant_upsert` - Storage time
- `num_patches` - Patch count per image

## Troubleshooting

### Service won't start

```bash
# Check Qdrant is running
docker ps | grep qdrant

# Check collections exist
python -m scripts.create_collection
```

### S3 access denied

```bash
# Verify AWS credentials
aws s3 ls s3://your-bucket/

# For IAM roles, ensure the role has s3:GetObject permission
```

### Out of memory (GPU)

```bash
# Reduce batch size
export BATCH_SIZE=16

# Or use CPU
export CUDA_VISIBLE_DEVICES=""
```

### Slow ingestion

- Use GPU if available
- Increase `BATCH_SIZE` for GPU
- Reduce `MAX_PATCHES_PER_IMAGE`
- Use larger stride (less overlap): `export PATCH_STRIDE=224`

## Repository Structure

```
askgvt-visual-ingest/
├── README.md
├── pyproject.toml
├── docker-compose.yml
├── scripts/
│   ├── create_collection.py    # Idempotent collection creation
│   └── smoke_ingest.py          # Smoke test script
├── src/
│   ├── config.py                # Environment-based configuration
│   ├── s3_client.py             # S3 URL parsing and fetching
│   ├── io_utils.py              # EXIF-safe image I/O
│   ├── patcher.py               # Sliding-window patch generation
│   ├── embedder.py              # OpenCLIP wrapper (image + text)
│   ├── qdrant_store.py          # Qdrant storage with multivector
│   ├── reranker.py              # MaxSim patch-level reranking
│   ├── search.py                # Search logic (text, image, hybrid)
│   ├── async_jobs.py            # Background job queue
│   └── service.py               # FastAPI application
└── tests/
    ├── test_http_ingest.py      # Ingestion tests with moto
    └── test_search.py           # Search and reranking tests
```

## Phase 2 Complete

**Implemented:**
✅ Text-to-image search
✅ Image-to-image search
✅ Patch-level MaxSim reranking
✅ Hybrid text-image search
✅ Async batch processing
✅ Advanced filtering
✅ Job tracking

**Future enhancements:**
- Object detection integration (GroundingDINO/SAM)
- Distributed processing (Celery/RabbitMQ)
- Vector quantization/compression
- Multi-modal reranking models
- Temporal search across video sequences

## License

MIT
