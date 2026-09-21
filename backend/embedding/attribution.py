"""IRIS query-time visual attribution — where in a 224x224 tile a query was matched.

ViT-B/32 sees a crop as a 7x7 grid of 32-px patches plus a CLS token. The image embedding IRIS searches with is the
CLS token after the last transformer block, and the last block's attention says how much each patch fed it. That is
the raw signal (step 3 of the recipe: CLS -> patch attention, 49 values). It does not depend on the query, so for a text
query the attention is weighted by its gradient (Chefer et al., ICCV 2021, "Generic Attention-model Explainability",
applied to the last layer only, with no rollout): the patches whose attention raises the similarity of the image to the
query count, the rest do not. If that gradient signal is degenerate (all zero), or there is no query, the raw attention
is used and the result says so.

The 7x7 grid is normalised to [0, 1], upsampled bilinearly to 224x224 and coloured transparent -> orange -> red as a
base64 RGBA PNG. Each patch covers about 320 m at 10 m/px, so this localises to patch level, not pixel level.

The attention weights are read with hooks on the last block's attention module. The model is shared with embedding jobs
running on other threads, so the hooks act only for the thread that installed them and are always removed.
"""

import base64
import io
import logging
import threading
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from embedding.embedder import RemoteCLIPEmbedder
from ingestion.loader import io_path

logger = logging.getLogger("iris.embedding.attribution")

GRID = 7  # patches per side (224 / 32)
SIZE = 224
METHOD_GRADIENT = "gradient_weighted_last_layer_attention"
METHOD_ATTENTION = "last_layer_cls_attention"

_lock = threading.Lock()  # one attribution at a time: each one temporarily hooks the shared model


def _patch_relevance(embedder: RemoteCLIPEmbedder, image: Image.Image, query: Optional[str]) -> Tuple[np.ndarray, str]:
    """(49 relevance values, method name) for one crop and, optionally, a text query."""
    model = embedder.model
    attn_module = model.visual.transformer.resblocks[-1].attn
    owner = threading.get_ident()
    captured: dict = {}

    def explicit_attention(module: torch.nn.MultiheadAttention, args: tuple, _kwargs: dict, _output: Any) -> Optional[tuple]:
        """Recompute this layer's self-attention step by step and hand back the same output.

        The fused attention kernel does not expose its weights, and the weights nn.MultiheadAttention returns on request
        are a view outside the autograd graph, so a gradient cannot be taken against them. Doing the arithmetic here puts
        the weights in the graph; the output is numerically the layer's own (checked in the tests).
        """
        if threading.get_ident() != owner:
            return None
        x = args[0]  # (batch, tokens, dim): the layer norm's output, since q = k = v in self-attention
        batch, tokens, dim = x.shape
        heads = module.num_heads
        q, k, v = F.linear(x, module.in_proj_weight, module.in_proj_bias).chunk(3, dim=-1)
        q, k, v = (t.reshape(batch, tokens, heads, dim // heads).transpose(1, 2) for t in (q, k, v))
        weights = ((q @ k.transpose(-2, -1)) * (dim // heads) ** -0.5).softmax(dim=-1)  # (batch, heads, tokens, tokens)
        context = (weights @ v).transpose(1, 2).reshape(batch, tokens, dim)
        captured["weights"] = weights
        return module.out_proj(context), None

    hook = attn_module.register_forward_hook(explicit_attention, with_kwargs=True)
    try:
        tensor = embedder.preprocess(image.convert("RGB")).unsqueeze(0).to(embedder.device)
        with torch.enable_grad():
            features = model.encode_image(tensor)
            weights = captured["weights"]
            raw = weights.detach().mean(dim=1)[0, 0, 1:]  # CLS row over the patch tokens, mean of the heads

            relevance, method = raw, METHOD_ATTENTION
            if query:
                text = torch.from_numpy(embedder.embed_text(query)).to(embedder.device)
                image_feat = features / features.norm(dim=-1, keepdim=True)
                score = (image_feat @ text.T).sum()
                (grad,) = torch.autograd.grad(score, weights)
                cam = (grad * weights).clamp(min=0).mean(dim=1)[0, 0, 1:].detach()
                if float(cam.max()) > 0.0:
                    relevance, method = cam, METHOD_GRADIENT
    finally:
        hook.remove()
    values = relevance.float().cpu().numpy()
    if values.size != GRID * GRID:
        raise ValueError(f"Expected {GRID * GRID} patch tokens, the model produced {values.size}")
    return values, method


def _colourise(grid: np.ndarray) -> Image.Image:
    """7x7 relevance -> 224x224 RGBA: transparent where the match is weak, orange then red where it is strong."""
    lo, hi = float(grid.min()), float(grid.max())
    norm = (grid - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(grid)
    tensor = torch.from_numpy(norm.astype(np.float32))[None, None]
    up = F.interpolate(tensor, size=(SIZE, SIZE), mode="bilinear", align_corners=False)[0, 0].clamp(0.0, 1.0).numpy()

    heat = np.clip((up - 0.35) / 0.65, 0.0, 1.0)  # 0 = orange, 1 = red
    rgba = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = (165.0 * (1.0 - heat)).astype(np.uint8)  # orange (255,165,0) -> red (255,0,0)
    rgba[..., 3] = (255.0 * np.clip((up - 0.1) / 0.9, 0.0, 1.0)).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def attribution_heatmap(crop_path: Path, query: Optional[str]) -> Tuple[str, str]:
    """Base64 RGBA PNG heatmap for one crop file, and the method that produced it."""
    embedder = RemoteCLIPEmbedder.get_instance()
    with Image.open(str(io_path(Path(crop_path)))) as opened:
        image = opened.convert("RGB")
    with _lock:
        values, method = _patch_relevance(embedder, image, query)
    buffer = io.BytesIO()
    _colourise(values.reshape(GRID, GRID)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii"), method
