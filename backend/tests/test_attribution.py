"""Tests for query-time attribution: the heatmap of where in a tile a query matched."""

import base64
import io
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from catalog.database import init_connection
from embedding import attribution
from embedding.embedder import RemoteCLIPEmbedder
from main import app
from s2_factory import Scenario, make_safe

NAME = "S2A_MSIL2A_20240110T050649_N0510_R019_T43PHM_20240110T090000"


@pytest.fixture
def scene(tmp_path):
    """One scene imported through the API: embedded, with its 224 px crops on disk."""
    client = TestClient(app)
    scn = Scenario(date="2024-01-10", noise_seed=1, landcover=[(40, 120, 40, 140, "built")])
    resp = client.post("/api/ingest", json={"file_path": str(make_safe(tmp_path / "src", NAME, scn))})
    assert resp.status_code == 200, resp.text
    conn = init_connection()
    try:
        tiles = {r[0]: r for r in conn.execute("SELECT tile_id, min_lon, min_lat, max_lon, max_lat FROM tiles WHERE scene_id = ?", (NAME,))}
    finally:
        conn.close()
    assert len(tiles) == 4
    return client, tiles


def decode(b64):
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def test_the_heatmap_is_a_transparent_224_png_over_the_tiles_bounds(scene):
    client, tiles = scene
    tile_id = next(t for t in tiles if t.endswith("crop_00000_00000"))
    resp = client.post("/api/search/attribution", json={"query": "buildings and roads", "tile_id": tile_id})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    image = decode(body["heatmap_base64"])
    assert image.format == "PNG" and image.mode == "RGBA" and image.size == (224, 224)
    alpha = np.asarray(image)[..., 3]
    assert alpha.min() == 0 and alpha.max() > 200  # transparent where the match is weak, strong where it is not
    assert 0.0 < (alpha > 0).mean() < 1.0

    assert body["tile_id"] == tile_id
    assert body["bounds_wgs84"] == pytest.approx(list(tiles[tile_id][1:]), abs=1e-9)
    corners = body["corners_wgs84"]
    assert len(corners) == 4 and all(len(c) == 2 for c in corners)
    lons, lats = [c[0] for c in corners], [c[1] for c in corners]
    b = body["bounds_wgs84"]
    assert b[0] - 1e-9 <= min(lons) and max(lons) <= b[2] + 1e-9 and b[1] - 1e-9 <= min(lats) and max(lats) <= b[3] + 1e-9
    assert corners[0][1] > corners[3][1]  # top-left is north of bottom-left
    assert body["method"] == "gradient_weighted_last_layer_attention" and body["query_conditioned"] is True


def test_different_queries_light_up_different_patches(scene):
    client, tiles = scene
    tile_id = next(t for t in tiles if t.endswith("crop_00000_00000"))
    maps = {}
    for query in ("buildings and roads", "dense green forest"):
        body = client.post("/api/search/attribution", json={"query": query, "tile_id": tile_id}).json()
        maps[query] = np.asarray(decode(body["heatmap_base64"]))[..., 3].astype(float)
    a, b = maps.values()
    assert not np.array_equal(a, b)
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] < 0.95


def test_bad_requests_are_rejected_with_clear_errors(scene):
    client, tiles = scene
    tile_id = next(iter(tiles))
    assert client.post("/api/search/attribution", json={"query": "x", "tile_id": "no_such_tile"}).status_code == 404
    assert client.post("/api/search/attribution", json={"query": "x", "tile_id": "../../etc/passwd"}).status_code == 422
    assert client.post("/api/search/attribution", json={"query": "", "tile_id": tile_id}).status_code == 422
    assert client.post("/api/search/attribution", json={"tile_id": tile_id}).status_code == 422
    assert client.post("/api/search/attribution", json={"query": "x" * 501, "tile_id": tile_id}).status_code == 422

    (Path("data/crops") / NAME / f"{tile_id}.png").unlink()
    resp = client.post("/api/search/attribution", json={"query": "x", "tile_id": tile_id})
    assert resp.status_code == 404 and "data" not in resp.json()["detail"].lower().replace("no longer on disk", "")  # no path in the message


# ---- the model-level guarantees ----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def embedder():
    return RemoteCLIPEmbedder.get_instance()


def a_crop():
    rng = np.random.default_rng(3)
    return Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))


def test_the_hook_returns_the_layers_own_output_and_is_removed(embedder):
    """Recomputing the last attention layer by hand must not change the embedding, or the heatmap would explain a
    different computation from the one search uses. And the shared model must be left untouched."""
    image = a_crop()
    attn = embedder.model.visual.transformer.resblocks[-1].attn
    with torch.no_grad():
        plain = embedder.model.encode_image(embedder.preprocess(image).unsqueeze(0).to(embedder.device))

    seen = {}
    real = embedder.model.encode_image

    def spy(x):
        features = real(x)
        seen["features"] = features.detach()  # a copy for the comparison; the caller still gets the graph
        return features

    embedder.model.encode_image = spy
    try:
        values, _ = attribution._patch_relevance(embedder, image, "airport")
    finally:
        embedder.model.encode_image = real

    assert torch.allclose(seen["features"], plain, atol=1e-4)
    assert len(attn._forward_hooks) == 0
    assert values.shape == (49,) and np.isfinite(values).all() and (values >= 0).all()


def test_without_a_query_the_raw_cls_attention_is_used(embedder):
    values, method = attribution._patch_relevance(embedder, a_crop(), None)
    assert method == attribution.METHOD_ATTENTION
    assert values.shape == (49,) and 0.0 < values.sum() <= 1.0  # attention over all 50 tokens sums to one


def test_the_query_changes_the_relevance(embedder):
    image = a_crop()
    a, method_a = attribution._patch_relevance(embedder, image, "an airport runway")
    b, method_b = attribution._patch_relevance(embedder, image, "a dense forest")
    assert method_a == method_b == attribution.METHOD_GRADIENT
    assert not np.allclose(a / a.max(), b / b.max())


def test_colouring_runs_from_transparent_through_orange_to_red():
    grid = np.zeros((7, 7), dtype=np.float32)
    grid[3, 3] = 1.0  # one hot patch
    grid[3, 4] = 0.5
    rgba = np.asarray(attribution._colourise(grid))
    assert rgba.shape == (224, 224, 4)
    assert rgba[0, 0, 3] == 0  # a cold corner is fully transparent
    hot = rgba[112, 112]  # centre of the hot patch
    assert hot[3] > 240 and hot[0] == 255 and hot[1] < 40  # opaque and red
    warm = rgba[112, 128 + 16]  # in the half-strength patch: orange-ish and partly transparent
    assert 0 < warm[3] < hot[3] and warm[1] > hot[1]
    assert (rgba[..., 0][rgba[..., 3] > 0] == 255).all()


def test_a_flat_relevance_grid_is_fully_transparent():
    assert np.asarray(attribution._colourise(np.full((7, 7), 0.3, dtype=np.float32)))[..., 3].max() == 0
