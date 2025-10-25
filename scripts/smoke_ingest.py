#!/usr/bin/env python3
"""
Smoke test for the visual ingest service.

Tests ingestion of a sample image from S3.

Usage:
    # Make sure service is running first:
    # uvicorn src.service:app --reload --port 8000

    python -m scripts.smoke_ingest --s3-url s3://my-bucket/sample.jpg --video-id 123 --frame-id 456

Or test with a pre-signed URL:
    python -m scripts.smoke_ingest --s3-url "https://bucket.s3.amazonaws.com/key.jpg" --video-id 123 --frame-id 456
"""

import argparse
import logging
import sys

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Run smoke test."""
    parser = argparse.ArgumentParser(description="Smoke test for visual ingest service")
    parser.add_argument(
        "--s3-url",
        required=True,
        help="S3 URL to ingest (s3:// or https://)",
    )
    parser.add_argument(
        "--video-id",
        type=int,
        required=True,
        help="Video ID",
    )
    parser.add_argument(
        "--frame-id",
        type=int,
        required=True,
        help="Frame ID",
    )
    parser.add_argument(
        "--service-url",
        default="http://localhost:8000",
        help="Service URL (default: http://localhost:8000)",
    )
    args = parser.parse_args()

    logger.info("=== Visual Ingest Smoke Test ===")
    logger.info(f"Service URL: {args.service_url}")
    logger.info(f"S3 URL: {args.s3_url}")
    logger.info(f"Video ID: {args.video_id}")
    logger.info(f"Frame ID: {args.frame_id}")

    # Check health
    logger.info("Checking service health...")
    try:
        with httpx.Client(timeout=30.0) as client:
            health_response = client.get(f"{args.service_url}/healthz")
            health_response.raise_for_status()
            health_data = health_response.json()

            if not health_data.get("ok"):
                logger.error("Service health check failed")
                logger.error(f"Health response: {health_data}")
                sys.exit(1)

            logger.info("✓ Service is healthy")
            logger.info(f"  - Qdrant reachable: {health_data.get('qdrant_reachable')}")
            logger.info(f"  - Model loaded: {health_data.get('model_loaded')}")
            logger.info(f"  - Using multivector: {health_data.get('using_multivector')}")

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        logger.error("Make sure the service is running: uvicorn src.service:app --port 8000")
        sys.exit(1)

    # Ingest frame
    logger.info("Ingesting frame...")
    try:
        with httpx.Client(timeout=60.0) as client:
            ingest_response = client.post(
                f"{args.service_url}/ingest/frame",
                json={
                    "s3_url": args.s3_url,
                    "video_id": args.video_id,
                    "frame_id": args.frame_id,
                },
            )
            ingest_response.raise_for_status()
            ingest_data = ingest_response.json()

            logger.info("✓ Frame ingested successfully")
            logger.info(f"  - Image ID: {ingest_data.get('image_id')}")
            logger.info(f"  - Global dim: {ingest_data.get('global_dim')}")
            logger.info(f"  - Num patches: {ingest_data.get('num_patches')}")
            logger.info(f"  - Patch dim: {ingest_data.get('patch_dim')}")
            logger.info(f"  - Multivector: {ingest_data.get('stored_multivector')}")

            durations = ingest_data.get("durations_ms", {})
            logger.info("  - Durations:")
            for key, value in durations.items():
                logger.info(f"    - {key}: {value:.1f}ms")

            logger.info("=== Smoke test PASSED ===")

    except httpx.HTTPStatusError as e:
        logger.error(f"Ingestion failed with HTTP {e.response.status_code}")
        logger.error(f"Response: {e.response.text}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
