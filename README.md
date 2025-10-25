# Visual Ingest Service

Production-ready HTTP service for ingesting images from S3 into Qdrant with OpenCLIP embeddings.

## Features

**What this service does:**
- Fetches images from S3 (supports both `s3://` URLs and HTTPS pre-signed URLs)
- Generates OpenCLIP embeddings:
  - **Global embedding** per image
  - **Patch embeddings** via advanced sliding-window with overlap, context padding, and optional multi-scale
- Stores vectors in Qdrant with multivector support (or fallback mode)
- Provides HTTP APIs for single and batch ingestion
- Production-ready with comprehensive error handling, logging, and monitoring

**What this service does NOT do (Phase 1):**
- No search/retrieval endpoints (storage only)
- No text embeddings or hybrid fusion
- No object detection (GroundingDINO/SAM)
- No async/background queues

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
│   ├── embedder.py              # OpenCLIP wrapper
│   ├── qdrant_store.py          # Qdrant storage with multivector
│   └── service.py               # FastAPI application
└── tests/
    └── test_http_ingest.py      # Comprehensive tests with moto
```

## Contributing

This is Phase 1 (storage only). Future phases may include:
- Search and retrieval endpoints
- Reranking with patch-level MaxSim
- Text-image hybrid search
- Object detection integration
- Async batch processing with queues

## License

MIT
