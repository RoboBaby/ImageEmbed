"""S3 client for fetching image bytes from S3 URLs."""

import re
from typing import Tuple
from urllib.parse import urlparse

import boto3
import httpx
from botocore.config import Config as BotoConfig

from src.config import Config


def parse_s3_url(url: str) -> Tuple[str, str]:
    """
    Parse S3 URL to extract bucket and key.

    Supports:
    - s3://bucket/key
    - https://bucket.s3.amazonaws.com/key
    - https://bucket.s3.region.amazonaws.com/key
    - https://s3.region.amazonaws.com/bucket/key

    Args:
        url: S3 URL string

    Returns:
        Tuple of (bucket, key)

    Raises:
        ValueError: If URL format is invalid
    """
    url = url.strip()

    # Handle s3:// protocol
    if url.startswith("s3://"):
        parts = url[5:].split("/", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"Invalid s3:// URL format: {url}")
        return parts[0], parts[1]

    # Handle HTTPS URLs (pre-signed or virtual-hosted style)
    if url.startswith("https://") or url.startswith("http://"):
        parsed = urlparse(url)
        hostname = parsed.hostname or ""

        # Virtual-hosted style: bucket.s3.region.amazonaws.com/key
        # or bucket.s3.amazonaws.com/key
        match = re.match(r"^([^.]+)\.s3[.-].*\.amazonaws\.com$", hostname)
        if match:
            bucket = match.group(1)
            key = parsed.path.lstrip("/")
            if not key:
                raise ValueError(f"No key found in URL: {url}")
            return bucket, key

        # Path style: s3.region.amazonaws.com/bucket/key
        if "s3" in hostname and "amazonaws.com" in hostname:
            path_parts = parsed.path.lstrip("/").split("/", 1)
            if len(path_parts) == 2 and path_parts[0] and path_parts[1]:
                return path_parts[0], path_parts[1]

        # Could be a pre-signed URL - try to extract from query params or path
        # For now, if we can't parse it, raise an error
        raise ValueError(f"Unable to parse S3 bucket and key from HTTPS URL: {url}")

    raise ValueError(f"Unsupported URL scheme. Use s3:// or https://: {url}")


def fetch_image_bytes(s3_url: str) -> bytes:
    """
    Fetch image bytes from an S3 URL.

    Supports both s3:// URLs (via boto3) and HTTPS pre-signed URLs (via httpx).

    Args:
        s3_url: S3 URL (s3:// or https://)

    Returns:
        Image bytes

    Raises:
        ValueError: If URL is invalid
        Exception: If fetch fails
    """
    s3_url = s3_url.strip()

    # Handle s3:// URLs with boto3
    if s3_url.startswith("s3://"):
        bucket, key = parse_s3_url(s3_url)

        boto_config = BotoConfig(
            connect_timeout=Config.S3_TIMEOUT_SECONDS,
            read_timeout=Config.S3_TIMEOUT_SECONDS,
        )

        s3_client = boto3.client(
            "s3",
            region_name=Config.AWS_REGION,
            endpoint_url=Config.S3_ENDPOINT_URL,
            config=boto_config,
        )

        try:
            response = s3_client.get_object(Bucket=bucket, Key=key)
            return response["Body"].read()
        except s3_client.exceptions.NoSuchKey:
            raise FileNotFoundError(f"S3 object not found: {s3_url}")
        except Exception as e:
            raise Exception(f"Failed to fetch from S3: {e}")

    # Handle HTTPS URLs (pre-signed)
    elif s3_url.startswith("https://") or s3_url.startswith("http://"):
        try:
            with httpx.Client(timeout=Config.S3_TIMEOUT_SECONDS) as client:
                response = client.get(s3_url)
                response.raise_for_status()
                return response.content
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise FileNotFoundError(f"Resource not found: {s3_url}")
            raise Exception(f"HTTP error fetching URL: {e}")
        except Exception as e:
            raise Exception(f"Failed to fetch from URL: {e}")

    else:
        raise ValueError(f"Unsupported URL scheme: {s3_url}")
