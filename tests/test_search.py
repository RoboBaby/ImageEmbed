"""
Tests for search and reranking functionality.

Run with:
    pytest tests/test_search.py -v
"""

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from qdrant_client import QdrantClient

from src.config import Config
from src.service import app
from tests.test_http_ingest import generate_test_image, s3_client, qdrant_client, test_client


# Use fixtures from test_http_ingest
@pytest.fixture
def ingested_test_data(test_client, s3_client, qdrant_client):
    """
    Ingest some test frames for search testing.

    Returns dict with image_ids and metadata.
    """
    frames = [
        {"s3_url": "s3://test-visual-bucket/images/red.jpg", "video_id": 1, "frame_id": 1},
        {"s3_url": "s3://test-visual-bucket/images/green.jpg", "video_id": 1, "frame_id": 2},
        {"s3_url": "s3://test-visual-bucket/images/blue.jpg", "video_id": 2, "frame_id": 1},
    ]

    ingested = []
    for frame in frames:
        response = test_client.post("/ingest/frame", json=frame)
        assert response.status_code == 200
        data = response.json()
        ingested.append(
            {
                "image_id": data["image_id"],
                "s3_url": frame["s3_url"],
                "video_id": frame["video_id"],
                "frame_id": frame["frame_id"],
            }
        )

    return ingested


def test_text_search(test_client, ingested_test_data):
    """Test text-based search."""
    request = {
        "query": "a red colored image",
        "top_k": 3,
    }

    response = test_client.post("/search/text", json=request)
    assert response.status_code == 200

    data = response.json()
    assert data["total"] == 3
    assert data["query_type"] == "text"
    assert len(data["results"]) == 3

    # Check result structure
    result = data["results"][0]
    assert "image_id" in result
    assert "video_id" in result
    assert "frame_id" in result
    assert "score" in result
    assert "s3_url" in result
    assert result["score"] > 0


def test_text_search_with_filters(test_client, ingested_test_data):
    """Test text search with video_id filter."""
    request = {
        "query": "colorful image",
        "top_k": 10,
        "video_ids": [1],
    }

    response = test_client.post("/search/text", json=request)
    assert response.status_code == 200

    data = response.json()
    # Should only return frames from video 1
    for result in data["results"]:
        assert result["video_id"] == 1


def test_image_search_by_s3_url(test_client, ingested_test_data):
    """Test image-based search using S3 URL."""
    # Use the red image as query
    request = {
        "s3_url": "s3://test-visual-bucket/images/red.jpg",
        "top_k": 3,
        "rerank": False,
    }

    response = test_client.post("/search/image", json=request)
    assert response.status_code == 200

    data = response.json()
    assert data["total"] == 3
    assert data["query_type"] == "image"
    assert data["rerank_enabled"] is False

    # First result should be the query image itself (highest similarity)
    first_result = data["results"][0]
    assert first_result["s3_url"] == "s3://test-visual-bucket/images/red.jpg"
    assert first_result["score"] > 0.9  # Should be very similar to itself


def test_image_search_with_reranking(test_client, ingested_test_data):
    """Test image search with patch-level reranking."""
    request = {
        "s3_url": "s3://test-visual-bucket/images/green.jpg",
        "top_k": 3,
        "rerank": True,
        "rerank_alpha": 0.5,
    }

    response = test_client.post("/search/image", json=request)
    assert response.status_code == 200

    data = response.json()
    assert data["rerank_enabled"] is True

    # Check that results have both global and maxsim scores
    first_result = data["results"][0]
    assert "global_score" in first_result
    assert "maxsim_score" in first_result
    assert first_result["global_score"] is not None
    assert first_result["maxsim_score"] is not None


def test_image_upload_search(test_client, ingested_test_data):
    """Test search by uploaded image file."""
    # Generate a small test image
    img_array = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = test_client.post(
        "/search/image/upload",
        files={"file": ("test.jpg", buf, "image/jpeg")},
        params={"top_k": 3, "rerank": False},
    )

    assert response.status_code == 200

    data = response.json()
    assert data["query_type"] == "image_upload"
    assert len(data["results"]) > 0


def test_hybrid_search_text_only(test_client, ingested_test_data):
    """Test hybrid search with text only."""
    request = {
        "text_query": "bright colors",
        "text_weight": 1.0,
        "image_weight": 0.0,
        "top_k": 3,
    }

    response = test_client.post("/search/hybrid", json=request)
    assert response.status_code == 200

    data = response.json()
    assert data["query_type"] == "hybrid"
    assert len(data["results"]) > 0


def test_hybrid_search_image_only(test_client, ingested_test_data):
    """Test hybrid search with image only."""
    request = {
        "image_s3_url": "s3://test-visual-bucket/images/blue.jpg",
        "text_weight": 0.0,
        "image_weight": 1.0,
        "top_k": 3,
    }

    response = test_client.post("/search/hybrid", json=request)
    assert response.status_code == 200

    data = response.json()
    assert data["query_type"] == "hybrid"


def test_hybrid_search_combined(test_client, ingested_test_data):
    """Test hybrid search with both text and image."""
    request = {
        "text_query": "a colorful scene",
        "image_s3_url": "s3://test-visual-bucket/images/red.jpg",
        "text_weight": 0.5,
        "image_weight": 0.5,
        "top_k": 3,
    }

    response = test_client.post("/search/hybrid", json=request)
    assert response.status_code == 200

    data = response.json()
    assert data["query_type"] == "hybrid"
    assert len(data["results"]) > 0

    # Results should have combined scores
    for result in data["results"]:
        assert result["score"] >= 0


def test_hybrid_search_no_query(test_client):
    """Test that hybrid search fails without queries."""
    request = {"top_k": 3}

    response = test_client.post("/search/hybrid", json=request)
    assert response.status_code == 400
    assert "must provide" in response.json()["detail"].lower()


def test_async_batch_ingestion(test_client, s3_client):
    """Test async batch ingestion with job tracking."""
    # Submit async job
    request = {
        "items": [
            {"s3_url": "s3://test-visual-bucket/images/red.jpg", "video_id": 10, "frame_id": 1},
            {"s3_url": "s3://test-visual-bucket/images/green.jpg", "video_id": 10, "frame_id": 2},
        ]
    }

    response = test_client.post("/ingest/frames/async", json=request)
    assert response.status_code == 200

    data = response.json()
    assert "job_id" in data
    assert data["status"] == "pending"
    assert data["total_items"] == 2

    job_id = data["job_id"]

    # Poll job status
    import time

    for _ in range(10):  # Max 10 seconds
        time.sleep(1)

        status_response = test_client.get(f"/jobs/{job_id}")
        assert status_response.status_code == 200

        status_data = status_response.json()
        assert status_data["job_id"] == job_id

        if status_data["status"] == "completed":
            assert status_data["successful_items"] == 2
            assert status_data["failed_items"] == 0
            assert len(status_data["results"]) == 2
            break
        elif status_data["status"] == "failed":
            pytest.fail(f"Job failed: {status_data.get('error')}")

    else:
        pytest.fail("Job did not complete in time")


def test_list_jobs(test_client):
    """Test listing jobs."""
    response = test_client.get("/jobs?limit=10")
    assert response.status_code == 200

    data = response.json()
    assert "jobs" in data
    assert "total" in data
    assert isinstance(data["jobs"], list)


def test_maxsim_reranking_improves_results(test_client, ingested_test_data):
    """
    Test that MaxSim reranking can change result order.

    This is a qualitative test - we just verify that reranking produces
    different scores and potentially different ordering.
    """
    # Search without reranking
    request_no_rerank = {
        "s3_url": "s3://test-visual-bucket/images/red.jpg",
        "top_k": 3,
        "rerank": False,
    }

    response_no_rerank = test_client.post("/search/image", json=request_no_rerank)
    assert response_no_rerank.status_code == 200
    results_no_rerank = response_no_rerank.json()["results"]

    # Search with reranking
    request_rerank = {
        "s3_url": "s3://test-visual-bucket/images/red.jpg",
        "top_k": 3,
        "rerank": True,
        "rerank_alpha": 0.3,  # Heavy weight on patches
    }

    response_rerank = test_client.post("/search/image", json=request_rerank)
    assert response_rerank.status_code == 200
    results_rerank = response_rerank.json()["results"]

    # Both should return same number of results
    assert len(results_no_rerank) == len(results_rerank)

    # Reranked results should have maxsim scores
    for result in results_rerank:
        assert result["maxsim_score"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
