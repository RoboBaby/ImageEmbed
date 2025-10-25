"""OpenCLIP embedding wrapper for images and patches."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import open_clip
import torch
from PIL import Image

from src.config import Config
from src.patcher import crop_patch, subsample_patches


class OpenCLIPEmbedder:
    """
    OpenCLIP model wrapper for generating L2-normalized embeddings.

    Supports:
    - Global image embeddings
    - Batch patch embeddings
    - Text embeddings (for text-image hybrid search)
    """

    def __init__(
        self,
        model_name: str,
        pretrained: str,
        device: Optional[str] = None,
    ):
        """
        Initialize the OpenCLIP embedder.

        Args:
            model_name: OpenCLIP model name (e.g., 'ViT-B-32')
            pretrained: Pretrained weights name (e.g., 'laion2b_s34b_b79k')
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_name = model_name
        self.pretrained = pretrained

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model and transforms
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        # Get tokenizer (not used for images, but available if needed)
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # Determine input size from preprocess transforms
        self.input_size = self._get_input_size()

    def _get_input_size(self) -> int:
        """Extract input size from preprocess transforms."""
        # The preprocess transform usually has Resize or CenterCrop
        # Default to 224 if we can't determine
        for transform in self.preprocess.transforms:
            if hasattr(transform, "size"):
                size = transform.size
                if isinstance(size, int):
                    return size
                elif isinstance(size, (list, tuple)) and len(size) > 0:
                    return size[0]
        return 224

    def embed_images(self, images: List[Image.Image]) -> np.ndarray:
        """
        Generate L2-normalized embeddings for a batch of images.

        Args:
            images: List of PIL Images

        Returns:
            Numpy array of shape [N, D] with L2-normalized embeddings (float32)
        """
        if not images:
            return np.array([], dtype=np.float32).reshape(0, self.model.visual.output_dim)

        batch_size = Config.BATCH_SIZE
        all_features = []

        with torch.inference_mode():
            for i in range(0, len(images), batch_size):
                batch = images[i : i + batch_size]

                # Preprocess images
                image_tensors = torch.stack([self.preprocess(img) for img in batch])
                image_tensors = image_tensors.to(self.device)

                # Encode
                features = self.model.encode_image(image_tensors)

                # L2 normalize
                features = features / features.norm(dim=-1, keepdim=True)

                # Move to CPU and convert to numpy
                all_features.append(features.cpu().numpy())

        # Concatenate all batches
        result = np.concatenate(all_features, axis=0).astype(np.float32)
        return result

    def embed_image(self, img: Image.Image) -> np.ndarray:
        """
        Generate L2-normalized embedding for a single image.

        Args:
            img: PIL Image

        Returns:
            Numpy array of shape [D] with L2-normalized embedding (float32)
        """
        result = self.embed_images([img])
        return result[0]

    def embed_patches(
        self, img: Image.Image, patch_specs: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, Dict[str, List]]:
        """
        Generate embeddings for image patches with metadata.

        Args:
            img: PIL Image
            patch_specs: List of patch specifications from patcher.compute_patches

        Returns:
            Tuple of:
            - embeddings: Numpy array [M, D] of L2-normalized embeddings (float32)
            - metadata: Dict with aligned arrays for patch_x, patch_y, patch_w, patch_h, patch_scale
        """
        # Apply max patches limit if configured
        if Config.MAX_PATCHES_PER_IMAGE and len(patch_specs) > Config.MAX_PATCHES_PER_IMAGE:
            patch_specs = subsample_patches(patch_specs, Config.MAX_PATCHES_PER_IMAGE)

        if not patch_specs:
            empty_emb = np.array([], dtype=np.float32).reshape(0, self.model.visual.output_dim)
            empty_meta = {
                "patch_x": [],
                "patch_y": [],
                "patch_w": [],
                "patch_h": [],
                "patch_scale": [],
            }
            return empty_emb, empty_meta

        # Crop patches using context boxes
        patch_images = [crop_patch(img, spec) for spec in patch_specs]

        # Embed patches
        embeddings = self.embed_images(patch_images)

        # Build aligned metadata arrays
        metadata = {
            "patch_x": [spec["x"] for spec in patch_specs],
            "patch_y": [spec["y"] for spec in patch_specs],
            "patch_w": [spec["w"] for spec in patch_specs],
            "patch_h": [spec["h"] for spec in patch_specs],
            "patch_scale": [spec["scale"] for spec in patch_specs],
        }

        return embeddings, metadata

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate L2-normalized embedding for a text query.

        Args:
            text: Text string

        Returns:
            Numpy array of shape [D] with L2-normalized embedding (float32)
        """
        result = self.embed_texts([text])
        return result[0]

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate L2-normalized embeddings for a batch of text queries.

        Args:
            texts: List of text strings

        Returns:
            Numpy array of shape [N, D] with L2-normalized embeddings (float32)
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.model.visual.output_dim)

        batch_size = Config.BATCH_SIZE
        all_features = []

        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]

                # Tokenize
                text_tokens = self.tokenizer(batch).to(self.device)

                # Encode
                features = self.model.encode_text(text_tokens)

                # L2 normalize
                features = features / features.norm(dim=-1, keepdim=True)

                # Move to CPU and convert to numpy
                all_features.append(features.cpu().numpy())

        # Concatenate all batches
        result = np.concatenate(all_features, axis=0).astype(np.float32)
        return result

    def get_embedding_dim(self) -> int:
        """Get the dimensionality of embeddings."""
        return self.model.visual.output_dim
