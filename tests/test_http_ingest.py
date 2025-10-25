"""
Comprehensive tests for HTTP ingestion service.

Tests S3 fetching (with moto), embedding, and Qdrant storage.

To run tests:
    # Start Qdrant first
    docker compose up -d

    # Run tests
    pytest tests/test_http_ingest.py -v

    # Or skip tests that require Qdrant
    pytest tests/test_http_ingest.py -v -m "not requires_qdrant"
"""

import io
import os
from typing import Generator

import boto3
import numpy as np
import pytest
from fastapi.testclient import TestClient
from moto import mock_s3
from PIL import Image
from qdrant_client import QdrantClient

from src.config import Config
from src.service import app

# Test configuration
TEST_BUCKET = "test-visual-bucket"
TEST_REGION = "us-east-1"


def generate_test_image(width: int = 400, height: int = 300, color: str = "red") -> bytes:
    """Generate a simple test image as JPEG bytes."""
    color_map = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
    }
    rgb = color_map.get(color, (128, 128, 128))

    # Create random image with dominant color
    img_array = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
    img_array[:, :, 0] += rgb[0]
    img_array[:, :, 1] += rgb[1]
    img_array[:, :, 2] += rgb[2]
    img_array = np.clip(img_array, 0, 255)

    img = Image.fromarray(img_array, mode="RGB")

    # Convert to JPEG bytes
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@pytest.fixture
def s3_client():
    """Create mocked S3 client and bucket with test images."""
    with mock_s3():
        # Set environment for boto3
        os.environ["AWS_ACCESS_KEY_ID"] = "testing"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
        os.environ["AWS_SECURITY_TOKEN"] = "testing"
        os.environ["AWS_SESSION_TOKEN"] = "testing"

        # Create S3 client and bucket
        s3 = boto3.client("s3", region_name=TEST_REGION)
        s3.create_bucket(Bucket=TEST_BUCKET)

        # Upload test images
        test_images = {
            "images/red.jpg": generate_test_image(400, 300, "red"),
            "images/green.jpg": generate_test_image(500, 400, "green"),
            "images/blue.jpg": generate_test_image(300, 300, "blue"),
        }

        for key, img_bytes in test_images.items():
            s3.put_object(Bucket=TEST_BUCKET, Key=key, Body=img_bytes)

        yield s3


@pytest.fixture
def qdrant_client() -> Generator[QdrantClient, None, None]:
    """
    Create Qdrant client for testing.

    Requires Qdrant to be running (docker compose up -d).
    """
    try:
        client = QdrantClient(url=Config.QDRANT_URL)
        client.get_collections()
        yield client
    except Exception as e:
        pytest.skip(f"Qdrant not available: {e}")


@pytest.fixture
def test_client() -> TestClient:
    """Create FastAPI test client."""
    return TestClient(app)


def test_health_check(test_client: TestClient):
    """Test health check endpoint."""
    response = test_client.get("/healthz")
    assert response.status_code == 200

    data = response.json()
    assert data["ok"] is True
    assert data["qdrant_reachable"] is True
    assert data["model_loaded"] is True
    assert "config" in data
    assert isinstance(data["using_multivector"], bool)


def test_ingest_single_frame_s3(test_client: TestClient, s3_client, qdrant_client: QdrantClient):
    """Test ingesting a single frame from S3."""
    # Prepare request
    request_data = {
        "s3_url": f"s3://{TEST_BUCKET}/images/red.jpg",
        "video_id": 123,
        "frame_id": 456,
        "timestamp_ms": 1500,
    }

    # Make request
    response = test_client.post("/ingest/frame", json=request_data)
    assert response.status_code == 200

    # Verify response
    data = response.json()
    assert data["image_id"] == "123:456"
    assert data["video_id"] == 123
    assert data["frame_id"] == 456
    assert data["global_dim"] > 0
    assert data["num_patches"] > 0
    assert data["patch_dim"] == data["global_dim"]
    assert isinstance(data["stored_multivector"], bool)

    # Verify durations
    durations = data["durations_ms"]
    assert "fetch" in durations
    assert "embed_global" in durations
    assert "embed_patches" in durations
    assert "qdrant_upsert" in durations
    assert "total" in durations

    # Verify storage in Qdrant
    image_id = data["image_id"]
    stored_multivector = data["stored_multivector"]

    if stored_multivector:
        # Multivector mode: single point with named vectors
        point = qdrant_client.retrieve(
            collection_name=Config.QDRANT_COLLECTION,
            ids=[image_id],
            with_vectors=True,
            with_payload=True,
        )[0]

        assert point.id == image_id
        assert "global" in point.vector
        assert "patches" in point.vector

        # Verify vectors
        global_vec = point.vector["global"]
        patches_vec = point.vector["patches"]

        assert len(global_vec) == data["global_dim"]
        assert len(patches_vec) == data["num_patches"]
        assert all(len(patch) == data["patch_dim"] for patch in patches_vec)

        # Verify payload
        payload = point.payload
        assert payload["video_id"] == 123
        assert payload["frame_id"] == 456
        assert payload["width"] == 400
        assert payload["height"] == 300
        assert payload["timestamp_ms"] == 1500
        assert len(payload["patch_x"]) == data["num_patches"]
        assert len(payload["patch_y"]) == data["num_patches"]
        assert len(payload["patch_w"]) == data["num_patches"]
        assert len(payload["patch_h"]) == data["num_patches"]
        assert len(payload["patch_scale"]) == data["num_patches"]

    else:
        # Fallback mode: separate collections
        global_point = qdrant_client.retrieve(
            collection_name=Config.QDRANT_COLLECTION,
            ids=[image_id],
            with_vectors=True,
            with_payload=True,
        )[0]

        assert global_point.id == image_id
        assert len(global_point.vector) == data["global_dim"]

        # Check patch collection
        patches_collection = f"{Config.QDRANT_COLLECTION}_patches"
        patch_points = qdrant_client.scroll(
            collection_name=patches_collection,
            scroll_filter={
                "must": [{"key": "frame_id", "match": {"value": image_id}}]
            },
            limit=100,
        )[0]

        assert len(patch_points) == data["num_patches"]


def test_ingest_missing_s3_object(test_client: TestClient, s3_client):
    """Test ingesting a non-existent S3 object."""
    request_data = {
        "s3_url": f"s3://{TEST_BUCKET}/images/missing.jpg",
        "video_id": 999,
        "frame_id": 999,
    }

    response = test_client.post("/ingest/frame", json=request_data)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_ingest_invalid_s3_url(test_client: TestClient):
    """Test ingesting with invalid S3 URL."""
    request_data = {
        "s3_url": "invalid://url",
        "video_id": 123,
        "frame_id": 456,
    }

    response = test_client.post("/ingest/frame", json=request_data)
    assert response.status_code in [400, 502]  # Bad request or bad gateway


def test_ingest_batch_frames(test_client: TestClient, s3_client, qdrant_client: QdrantClient):
    """Test batch ingestion."""
    request_data = {
        "items": [
            {
                "s3_url": f"s3://{TEST_BUCKET}/images/red.jpg",
                "video_id": 1,
                "frame_id": 101,
            },
            {
                "s3_url": f"s3://{TEST_BUCKET}/images/green.jpg",
                "video_id": 1,
                "frame_id": 102,
            },
            {
                "s3_url": f"s3://{TEST_BUCKET}/images/missing.jpg",  # This will fail
                "video_id": 1,
                "frame_id": 103,
            },
        ]
    }

    response = test_client.post("/ingest/frames", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert data["total"] == 3
    assert data["successful"] == 2
    assert data["failed"] == 1
    assert len(data["results"]) == 3

    # Check individual results
    results = data["results"]
    assert results[0]["ok"] is True
    assert results[0]["image_id"] == "1:101"
    assert results[0]["num_patches"] > 0

    assert results[1]["ok"] is True
    assert results[1]["image_id"] == "1:102"
    assert results[1]["num_patches"] > 0

    assert results[2]["ok"] is False
    assert "error" in results[2]


def test_embeddings_are_normalized(test_client: TestClient, s3_client, qdrant_client: QdrantClient):
    """Test that embeddings are L2-normalized."""
    request_data = {
        "s3_url": f"s3://{TEST_BUCKET}/images/blue.jpg",
        "video_id": 555,
        "frame_id": 777,
    }

    response = test_client.post("/ingest/frame", json=request_data)
    assert response.status_code == 200

    data = response.json()
    image_id = data["image_id"]
    stored_multivector = data["stored_multivector"]

    if stored_multivector:
        point = qdrant_client.retrieve(
            collection_name=Config.QDRANT_COLLECTION,
            ids=[image_id],
            with_vectors=True,
        )[0]

        # Check global vector normalization
        global_vec = np.array(point.vector["global"])
        global_norm = np.linalg.norm(global_vec)
        assert abs(global_norm - 1.0) < 1e-5, f"Global vector not normalized: {global_norm}"

        # Check patch vector normalization
        patches_vec = np.array(point.vector["patches"])
        for i, patch_vec in enumerate(patches_vec):
            patch_norm = np.linalg.norm(patch_vec)
            assert abs(patch_norm - 1.0) < 1e-5, f"Patch {i} not normalized: {patch_norm}"

    else:
        # Check global
        global_point = qdrant_client.retrieve(
            collection_name=Config.QDRANT_COLLECTION,
            ids=[image_id],
            with_vectors=True,
        )[0]

        global_vec = np.array(global_point.vector)
        global_norm = np.linalg.norm(global_vec)
        assert abs(global_norm - 1.0) < 1e-5


def test_patch_generation(test_client: TestClient, s3_client):
    """Test that patches are generated according to configuration."""
    request_data = {
        "s3_url": f"s3://{TEST_BUCKET}/images/red.jpg",
        "video_id": 222,
        "frame_id": 333,
    }

    response = test_client.post("/ingest/frame", json=request_data)
    assert response.status_code == 200

    data = response.json()

    # Should have at least one patch
    assert data["num_patches"] >= 1

    # Patch dim should match global dim
    assert data["patch_dim"] == data["global_dim"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
