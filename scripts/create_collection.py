#!/usr/bin/env python3
"""
Idempotent script to create Qdrant collections for visual ingest.

Usage:
    python -m scripts.create_collection
"""

import logging
import sys

import open_clip
from qdrant_client import QdrantClient

from src.config import Config
from src.qdrant_store import ensure_collections

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Create Qdrant collections."""
    logger.info("=== Qdrant Collection Creation ===")
    logger.info(f"Qdrant URL: {Config.QDRANT_URL}")
    logger.info(f"Collection: {Config.QDRANT_COLLECTION}")
    logger.info(f"Use multivector: {Config.USE_QDRANT_MULTIVECTOR}")
    logger.info(f"Model: {Config.OPENCLIP_MODEL} ({Config.OPENCLIP_PRETRAINED})")

    # Connect to Qdrant
    try:
        client = QdrantClient(
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_API_KEY,
        )
        logger.info("Connected to Qdrant")
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        logger.error("Make sure Qdrant is running (docker compose up -d)")
        sys.exit(1)

    # Determine embedding dimension from model
    try:
        logger.info("Loading model to determine embedding dimension...")
        model, _, _ = open_clip.create_model_and_transforms(
            Config.OPENCLIP_MODEL,
            pretrained=Config.OPENCLIP_PRETRAINED,
        )
        dim = model.visual.output_dim
        logger.info(f"Embedding dimension: {dim}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    # Create collections
    try:
        using_multivector = ensure_collections(
            client=client,
            use_multivector=Config.USE_QDRANT_MULTIVECTOR,
            dim=dim,
            collection=Config.QDRANT_COLLECTION,
        )

        if using_multivector:
            logger.info(f"✓ Collection '{Config.QDRANT_COLLECTION}' ready (multivector mode)")
        else:
            logger.info(
                f"✓ Collections '{Config.QDRANT_COLLECTION}' and "
                f"'{Config.QDRANT_COLLECTION}_patches' ready (fallback mode)"
            )

        # Verify collections exist
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        logger.info(f"Existing collections: {collection_names}")

        logger.info("=== Collection creation complete ===")

    except Exception as e:
        logger.error(f"Failed to create collections: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
