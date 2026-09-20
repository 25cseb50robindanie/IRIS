"""Tests for step-by-step import progress (GET /api/pipeline/{id})."""

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import api.ingest as ingest_module
from api.pipeline_status import PipelineTracker
from main import app
from s2_factory import Scenario, make_safe

NAME_A = "S2A_MSIL2A_20240110T050649_N0510_R019_T43PHM_20240110T090000"
NAME_B = "S2A_MSIL2A_20240214T050649_N0510_R019_T43PHM_20240214T090000"
PATCH = (200, 260, 300, 380, 0.35)


def steps_by_key(snapshot):
    return {s["key"]: s for s in snapshot["steps"]}


def test_tracker_records_planned_steps_progress_and_completion():
    t = PipelineTracker()
    t.start("pipeline-0001", "scene.SAFE")
    t.complete("pipeline-0001", "detect", "Sentinel-2 L2A detected")
    t.plan("pipeline-0001", ["bands", "cog", "crops"])

    snap = t.snapshot("pipeline-0001")
    assert [s["state"] for s in snap["steps"]] == ["done", "pending", "pending", "pending"]

    t.progress("pipeline-0001", "bands", 512, 1024, "50%")
    step = steps_by_key(t.snapshot("pipeline-0001"))["bands"]
    assert (step["state"], step["done"], step["total"], step["detail"]) == ("active", 512, 1024, "50%")

    t.set_scene("pipeline-0001", "scene_x")
    t.finish("pipeline-0001", "Complete")
    by_scene = t.snapshot("scene_x")  # also addressable by the scene it produced
    assert by_scene["state"] == "done" and by_scene["message"] == "Complete"
    assert [s["state"] for s in by_scene["steps"]] == ["done", "skipped", "skipped", "skipped"]  # never reached


def test_tracker_marks_the_running_step_failed_and_ends_the_pipeline():
    t = PipelineTracker()
    t.start("pipeline-0002")
    t.plan("pipeline-0002", ["cog"])
    t.complete("pipeline-0002", "detect")
    t.begin("pipeline-0002", "cog")
    assert t.fail_active("pipeline-0002", "disk full") is True

    snap = t.snapshot("pipeline-0002")
    assert snap["state"] == "failed" and snap["message"] == "Import failed — disk full"
    assert steps_by_key(snap)["cog"]["state"] == "failed"
    t.finish("pipeline-0002", "Complete")  # a failed import is never relabelled as complete
    assert t.snapshot("pipeline-0002")["state"] == "failed"


def test_unknown_pipeline_is_404_and_bad_id_is_rejected(tmp_path):
    client = TestClient(app)
    assert client.get("/api/pipeline/does-not-exist").status_code == 404
    resp = client.post("/api/ingest", json={"file_path": str(tmp_path), "pipeline_id": "short"})
    assert resp.status_code == 422  # ids are validated against an allow-list pattern


def test_safe_import_reports_every_step_and_the_final_summary(tmp_path):
    client = TestClient(app)
    safe = make_safe(tmp_path / "src", NAME_A, Scenario(date="2024-01-10"))

    resp = client.post("/api/ingest", json={"file_path": str(safe), "pipeline_id": "import-safe-0001"})
    assert resp.status_code == 200
    assert resp.json()["pipeline_id"] == "import-safe-0001"

    snap = client.get("/api/pipeline/import-safe-0001").json()
    assert snap["state"] == "done"
    assert [s["key"] for s in snap["steps"]] == [
        "detect", "bands", "scl", "rgb", "cog", "crops", "embed", "index", "overlap", "change",
    ]
    steps = steps_by_key(snap)
    assert steps["detect"]["detail"] == "Sentinel-2 L2A detected"
    assert steps["bands"]["label"] == "Extracting bands (B02, B03, B04, B08)"
    assert all(steps[k]["state"] == "done" for k in ("detect", "bands", "scl", "rgb", "cog", "crops", "embed", "index", "overlap"))
    assert steps["crops"]["detail"] == "4 crops"
    assert steps["embed"]["detail"].startswith("4 crops on ")
    assert steps["index"]["detail"] == "4 tiles"
    # a first scene has nothing to compare with: change detection is skipped, and the analyst is told what enables it
    assert steps["change"]["state"] == "skipped"
    assert "Import a second scene of the same area" in steps["change"]["detail"]
    assert snap["message"] == "Complete — 4 tiles indexed, 0 change candidates found"
    assert snap["scene_id"] == NAME_A
    assert client.get(f"/api/pipeline/{NAME_A}").json()["pipeline_id"] == "import-safe-0001"


def test_second_scene_reports_change_detection_against_the_first(tmp_path):
    client = TestClient(app)
    a = make_safe(tmp_path / "src", NAME_A, Scenario(date="2024-01-10", noise_seed=1))
    b = make_safe(tmp_path / "src", NAME_B, Scenario(date="2024-02-14", noise_seed=2, patches=[PATCH]))
    client.post("/api/ingest", json={"file_path": str(a), "pipeline_id": "import-first-01"})
    client.post("/api/ingest", json={"file_path": str(b), "pipeline_id": "import-second-1"})

    snap = client.get("/api/pipeline/import-second-1").json()
    steps = steps_by_key(snap)
    assert snap["state"] == "done"
    assert steps["overlap"]["detail"] == "1 overlapping scene"
    assert steps["change"]["state"] == "done"
    assert steps["change"]["detail"] == f"against {NAME_A}: 1 candidate"
    assert snap["message"] == "Complete — 4 tiles indexed, 1 change candidate found"


def test_single_file_import_has_no_safe_only_steps(tmp_path):
    client = TestClient(app)
    tif = tmp_path / "S2A_MSIL2A_20240110T053651_plain.tif"
    with rasterio.open(
        tif, "w", driver="GTiff", height=256, width=256, count=3, dtype="uint8", crs="EPSG:32643",
        transform=from_origin(300000, 3100000, 10, 10), nodata=0,
    ) as dst:
        dst.write(np.random.randint(1, 255, (3, 256, 256), dtype=np.uint8))

    client.post("/api/ingest", json={"file_path": str(tif), "pipeline_id": "import-plain-01"})
    snap = client.get("/api/pipeline/import-plain-01").json()
    assert [s["key"] for s in snap["steps"]] == ["detect", "cog", "crops", "embed", "index", "overlap", "change"]
    assert snap["steps"][0]["detail"].endswith("raster detected")


def test_failed_import_marks_the_step_and_ends_the_pipeline(tmp_path):
    client = TestClient(app)
    (tmp_path / "not_a_product").mkdir()
    resp = client.post("/api/ingest", json={"file_path": str(tmp_path / "not_a_product"), "pipeline_id": "import-fail-001"})
    assert resp.status_code == 400

    snap = client.get("/api/pipeline/import-fail-001").json()
    assert snap["state"] == "failed"
    assert steps_by_key(snap)["detect"]["state"] == "failed"
    assert "Sentinel-2" in snap["message"]


def test_missing_checkpoint_fails_the_embed_step_but_the_import_completes(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest_module, "find_checkpoint", lambda *a, **k: None)
    client = TestClient(app)
    safe = make_safe(tmp_path / "src", NAME_A, Scenario(date="2024-01-10"))
    client.post("/api/ingest", json={"file_path": str(safe), "pipeline_id": "import-nockpt-1"})

    snap = client.get("/api/pipeline/import-nockpt-1").json()
    steps = steps_by_key(snap)
    assert steps["embed"]["state"] == "failed" and "RemoteCLIP-ViT-B-32.pt" in steps["embed"]["detail"]
    assert steps["cog"]["state"] == "done"  # the scene is still imported and viewable
    assert snap["state"] == "done" and snap["message"].startswith("Completed with errors")
