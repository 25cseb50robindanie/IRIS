"""IRIS RemoteCLIP Embedder — Offline inference for image crops and text queries."""

import logging
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import open_clip
import torch
from PIL import Image

from ingestion.loader import io_path

logger = logging.getLogger("iris.embedding.embedder")

DEFAULT_CHECKPOINT_PATHS = [
    Path("checkpoints/RemoteCLIP-ViT-B-32.pt"),
    Path("../checkpoints/RemoteCLIP-ViT-B-32.pt"),
    Path(__file__).resolve().parent.parent.parent / "checkpoints" / "RemoteCLIP-ViT-B-32.pt",
]


CHECKPOINT_MISSING_MSG = (
    "RemoteCLIP checkpoint not found. Expected checkpoints/RemoteCLIP-ViT-B-32.pt "
    "(searched relative to the working directory and the project root). "
    "IRIS is offline-only and never downloads weights at runtime; copy the file into place and retry."
)


def find_checkpoint(path: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """Return the first existing RemoteCLIP checkpoint (explicit path first, then defaults), or None."""
    if path:
        p = Path(path)
        if p.exists():
            return p.resolve()

    for candidate in DEFAULT_CHECKPOINT_PATHS:
        if candidate.exists():
            return candidate.resolve()
    return None


class RemoteCLIPEmbedder:
    """Manages RemoteCLIP ViT-B/32 model loading and vector inference."""

    _instance: Optional["RemoteCLIPEmbedder"] = None

    def __init__(self, checkpoint_path: Optional[Union[str, Path]] = None, model_name: str = "ViT-B-32") -> None:
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device for RemoteCLIP inference: %s", self.device)

        # 1. Resolve local checkpoint path
        resolved_ckpt = find_checkpoint(checkpoint_path)
        if resolved_ckpt is None:
            logger.error(CHECKPOINT_MISSING_MSG)
            raise FileNotFoundError(CHECKPOINT_MISSING_MSG)

        logger.info("Loading RemoteCLIP model %s from local checkpoint: %s", self.model_name, resolved_ckpt)

        # 2. Initialize OpenCLIP architecture (no internet pretrained weights)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.model_name)
        self.tokenizer = open_clip.get_tokenizer(self.model_name)

        # 3. Load checkpoint state dict
        checkpoint = torch.load(str(resolved_ckpt), map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

        logger.info("RemoteCLIP %s initialized and ready for inference", self.model_name)

    @classmethod
    def get_instance(cls, checkpoint_path: Optional[Union[str, Path]] = None) -> "RemoteCLIPEmbedder":
        """Singleton accessor to avoid re-loading weights into memory repeatedly."""
        if cls._instance is None:
            cls._instance = cls(checkpoint_path=checkpoint_path)
        return cls._instance

    def embed_images(
        self,
        images: List[Union[Image.Image, Path, str]],
        batch_size: int = 32,
        on_batch: Optional[Callable[[int, int], None]] = None,
    ) -> np.ndarray:
        """Compute L2-normalized 512-dim embedding vectors for a list of images.

        Args:
            images: List of PIL Images or image file paths.
            batch_size: Batch size for forward pass (default 32).
            on_batch: Optional callback(done, total) invoked after each batch.

        Returns:
            np.ndarray of shape (N, 512), dtype float32, normalized to unit length.
            Rows are never patched up: a non-finite row is returned as-is so the caller
            can drop that crop rather than index a fabricated vector.
        """
        if not images:
            return np.empty((0, 512), dtype=np.float32)

        all_embeddings: List[np.ndarray] = []

        for i in range(0, len(images), batch_size):
            batch_items = images[i : i + batch_size]
            tensors: List[torch.Tensor] = []

            for item in batch_items:
                if isinstance(item, (str, Path)):
                    img = Image.open(str(io_path(Path(item)))).convert("RGB")
                elif isinstance(item, Image.Image):
                    img = item.convert("RGB")
                else:
                    raise TypeError(f"Unsupported image type: {type(item)}")

                tensors.append(self.preprocess(img))

            batch_tensor = torch.stack(tensors).to(self.device)

            with torch.no_grad():
                features = self.model.encode_image(batch_tensor)
                # L2 normalize
                features = features / features.norm(dim=-1, keepdim=True)
                emb = features.cpu().numpy().astype(np.float32)

            if not np.isfinite(emb).all():
                logger.warning("Non-finite values in image embedding batch starting at %d", i)

            all_embeddings.append(emb)
            if on_batch is not None:
                on_batch(min(i + batch_size, len(images)), len(images))

        result = np.vstack(all_embeddings)
        return result

    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """Compute L2-normalized 512-dim embedding vector(s) for text queries.
        
        Args:
            text: Query string or list of query strings.
            
        Returns:
            np.ndarray of shape (N, 512), dtype float32, normalized to unit length.
        """
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        tokens = self.tokenizer(texts).to(self.device)

        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            emb = features.cpu().numpy().astype(np.float32)

        if not np.isfinite(emb).all():
            raise ValueError("RemoteCLIP produced a non-finite text embedding")

        return emb
