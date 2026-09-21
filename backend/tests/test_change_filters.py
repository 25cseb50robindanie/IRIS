"""Tests for confidence scoring, candidate noise reduction, the filtered candidate API, and multi-date pairing."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from catalog import changes
from catalog.database import init_connection
from change_detection import params, run_change_detection
from change_detection.direction import CHANGE_TYPES
from main import app
from s2_factory import Scenario, ingest_without_embedding, make_safe

@pytest.fixture(autouse=True)
def no_large_area_merge(monkeypatch):
    """These fixtures put patches 40 px apart, which the large-area pass would (rightly) merge into one detection.
    They test scoring, filtering and caps blob by blob, so the wide pass is off here (tests/test_grouping.py covers it)."""
    monkeypatch.setattr(params, "LARGE_AREA_MIN_BLOBS", 10**6)
    monkeypatch.setattr(params, "LARGE_AREA_MIN_PIXELS", 10**12)


NAME_A = "S2A_MSIL2A_20240110T050649_N0510_R019_T43PHM_20240110T090000"
NAME_B = "S2A_MSIL2A_20240214T050649_N0510_R019_T43PHM_20240214T090000"
NAME_C = "S2A_MSIL2A_20240320T050649_N0510_R019_T43PHM_20240320T090000"

# Changes of clearly different strength (NIR factor: lower = stronger drop), each 100 x 140 px, plus a small real one
STRENGTHS = [
    (60, 160, 60, 200, 0.25),  # very strong loss
    (200, 300, 60, 200, 0.5),  # strong loss
    (340, 440, 60, 200, 0.7),  # moderate loss
    (60, 160, 300, 400, 1.35),  # gain
]


def centre_of(patch):
    r0, r1, c0, c1, _ = patch
    return 300000.0 + 10.0 * (c0 + c1) / 2, 3100000.0 - 10.0 * (r0 + r1) / 2  # native x, y of the patch centre


def candidate_at(cands, patch):
    x, y = centre_of(patch)
    inside = [c for c in cands if c["min_x"] <= x <= c["max_x"] and c["min_y"] <= y <= c["max_y"]]
    return max(inside, key=lambda c: c["area_px"]) if inside else None


def make_pair(tmp_path, patches, name_b=NAME_B, **b_kwargs):
    a = ingest_without_embedding(make_safe(tmp_path / "src", NAME_A, Scenario(date="2024-01-10", noise_seed=1)))
    b = ingest_without_embedding(
        make_safe(tmp_path / "src", name_b, Scenario(date="2024-02-14", noise_seed=2, gain=1.04, offset=15, patches=patches, **b_kwargs))
    )
    result = run_change_detection(a, b)
    conn = init_connection()
    job = changes.get_job_for_pair(conn, a, b)
    cands = changes.list_candidates(conn)
    conn.close()
    return result, job, cands


def test_confidence_spreads_across_blobs_instead_of_saturating(tmp_path):
    result, job, cands = make_pair(tmp_path, STRENGTHS)
    assert result["status"] == "completed"

    confs = [c["confidence"] for c in cands]
    pcts = [c["norm_cluster_dist"] for c in cands]
    assert len(cands) >= 4
    assert max(confs) < 1.0 and min(confs) >= params.MIN_STORED_CONFIDENCE
    assert max(confs) - min(confs) > 0.2, f"confidence barely varies: {sorted(confs)}"
    # the cluster-distance term is a rank among the blobs, so it must actually use its range
    assert min(pcts) < 0.2 and max(pcts) > 0.8, sorted(pcts)
    assert len(set(round(p, 3) for p in pcts)) == len(pcts)  # no ties across distinct blobs

    # alignment for a same-tile pair is 0.90 (good, not measured), terrain is the documented placeholder
    assert {c["alignment_quality"] for c in cands} == {params.SAME_TILE_ALIGNMENT_QUALITY} == {0.9}
    assert {c["terrain_flatness"] for c in cands} == {1.0}
    # valid coverage is per blob and stays in [0, 1]
    assert all(0.3 <= c["valid_coverage"] <= 1.0 for c in cands)

    # stronger change -> higher confidence, and an obvious change scores well up the range
    strongest, strong, moderate = (candidate_at(cands, p) for p in STRENGTHS[:3])
    assert strongest and strong and moderate
    assert strongest["confidence"] > strong["confidence"] > moderate["confidence"]
    assert 0.7 <= strongest["confidence"] <= 0.97


@pytest.mark.parametrize(
    "patch, above_default",
    [
        ((60, 100, 60, 100, 0.25), True),  # small change: ECC correlation stays high, and outranks the 0.90 default
        ((60, 160, 60, 200, 0.25), False),  # a big real change lowers rho, which is reported as measured (still > 0.8)
    ],
)
def test_ecc_alignment_quality_is_the_measured_rho(tmp_path, patch, above_default):
    scene_b = Scenario(date="2024-02-14", noise_seed=2, patches=[patch], shift=(1.2, -0.7))
    a = ingest_without_embedding(make_safe(tmp_path / "src", NAME_A, Scenario(date="2024-01-10", noise_seed=1)))
    b = ingest_without_embedding(make_safe(tmp_path / "src", NAME_B.replace("T43PHM", "T43PHN"), scene_b))
    run_change_detection(a, b)
    conn = init_connection()
    cands = changes.list_candidates(conn)
    job = changes.get_job_for_pair(conn, a, b)
    conn.close()

    assert job["details"]["alignment"]["method"] == "ecc"
    rho = job["details"]["alignment"]["rho"]
    assert rho > 0.8  # passed the gate
    # the stored quality is the measured correlation, never a default; "higher than 0.90" must be earned
    assert cands and all(c["alignment_quality"] == pytest.approx(rho, abs=1e-3) for c in cands)
    assert (rho > params.SAME_TILE_ALIGNMENT_QUALITY) is above_default


def test_minimum_mapping_unit_is_one_hectare_and_a_parameter(tmp_path, monkeypatch):
    assert params.MIN_BLOB_PIXELS == 100  # 1 ha at 10 m
    big = (100, 112, 100, 112, 0.2)  # 144 px: above the unit
    small = (300, 309, 300, 310, 0.2)  # 90 px: below it, i.e. field-boundary noise
    _, job, cands = make_pair(tmp_path, [big, small])

    assert cands and all(c["area_px"] >= params.MIN_BLOB_PIXELS for c in cands)
    assert candidate_at(cands, big) is not None
    assert candidate_at(cands, small) is None
    assert job["details"]["detection"]["min_blob_pixels"] == params.MIN_BLOB_PIXELS


def test_the_minimum_mapping_unit_is_read_from_the_config_not_hardcoded(tmp_path, monkeypatch):
    monkeypatch.setattr(params, "MIN_BLOB_PIXELS", 60)  # now the 90 px patch is large enough to keep
    small = (300, 309, 300, 310, 0.2)
    _, job, cands = make_pair(tmp_path, [(100, 112, 100, 112, 0.2), small])

    assert candidate_at(cands, small) is not None
    assert job["details"]["detection"]["min_blob_pixels"] == 60


def test_candidates_below_the_confidence_threshold_are_not_stored(tmp_path, monkeypatch):
    monkeypatch.setattr(params, "MIN_STORED_CONFIDENCE", 0.85)
    _, job, cands = make_pair(tmp_path, STRENGTHS)

    scoring = job["details"]["scoring"]
    assert scoring["min_stored_confidence"] == 0.85
    assert scoring["dropped_low_confidence"] > 0
    assert all(c["confidence"] >= 0.85 for c in cands)
    assert scoring["candidates"] == len(cands)
    assert scoring["candidates_found"] == len(cands) + scoring["dropped_low_confidence"]  # nothing vanishes silently


def make_three_scene_archive(tmp_path):
    """A (Jan), B (Feb, with changes), C (Mar, with different changes), imported in date order through the API."""
    client = TestClient(app)
    scenes = [
        (NAME_A, Scenario(date="2024-01-10", noise_seed=1)),
        (NAME_B, Scenario(date="2024-02-14", noise_seed=2, patches=STRENGTHS[:2])),
        (NAME_C, Scenario(date="2024-03-20", noise_seed=3, patches=STRENGTHS[:2] + STRENGTHS[3:])),
    ]
    for name, scn in scenes:
        assert client.post("/api/ingest", json={"file_path": str(make_safe(tmp_path / "src", name, scn))}).status_code == 200
    return client


def test_api_filters_sort_cap_and_counts(tmp_path):
    client = TestClient(app)
    _, _, _ = make_pair(tmp_path, STRENGTHS)
    everything = client.get("/api/changes").json()

    assert everything["total"] == everything["matching"] == everything["shown"] == len(everything["candidates"]) >= 4
    assert everything["display_cap"] == 200 and everything["found"] >= everything["total"]

    # confidence slider: only what is at or above it, with the excluded count reported
    cut = sorted(c["confidence"] for c in everything["candidates"])[2]
    high = client.get("/api/changes", params={"min_confidence": cut}).json()
    assert high["matching"] < everything["matching"]
    assert all(c["confidence"] >= cut for c in high["candidates"])
    assert high["excluded_by_confidence"] >= everything["matching"] - high["matching"]
    assert high["filtered_by_confidence"] is True and high["total"] == everything["total"]

    # change types
    kinds = sorted({c["change_type"] for c in everything["candidates"]})
    assert kinds and set(kinds) <= set(CHANGE_TYPES)
    only = client.get("/api/changes", params={"types": kinds[0]}).json()
    assert only["matching"] >= 1 and {c["change_type"] for c in only["candidates"]} == {kinds[0]}
    both = client.get("/api/changes", params={"types": f"{kinds[0]},construction"}).json()
    assert {c["change_type"] for c in both["candidates"]} <= {kinds[0], "construction"}
    assert client.get("/api/changes", params={"types": "vegetation_loss"}).status_code == 400  # the old names are gone
    assert client.get("/api/changes", params={"directions": "sideways"}).status_code == 400
    named = sorted({c["direction"] for c in everything["candidates"]})
    by_direction = client.get("/api/changes", params={"directions": named[0]}).json()
    assert by_direction["matching"] >= 1 and {c["direction"] for c in by_direction["candidates"]} == {named[0]}

    # sorting changes the order, never the membership
    by_area = client.get("/api/changes", params={"sort": "area"}).json()["candidates"]
    assert [c["area_px"] for c in by_area] == sorted((c["area_px"] for c in by_area), reverse=True)
    assert {c["candidate_id"] for c in by_area} == {c["candidate_id"] for c in everything["candidates"]}
    by_conf = client.get("/api/changes", params={"sort": "confidence"}).json()["candidates"]
    assert [c["confidence"] for c in by_conf] == sorted((c["confidence"] for c in by_conf), reverse=True)

    # per-pair cap keeps the STRONGEST, and the totals still say how many exist
    capped = client.get("/api/changes", params={"limit": 2}).json()
    assert capped["shown"] == 2 and capped["matching"] == everything["matching"] and capped["display_cap"] == 2
    assert {c["candidate_id"] for c in capped["candidates"]} == {c["candidate_id"] for c in by_conf[:2]}


def test_third_scene_pairs_with_its_nearest_prior_and_keeps_earlier_results(tmp_path):
    client = make_three_scene_archive(tmp_path)
    body = client.get("/api/changes").json()

    pairs = {(p["scene_a_date"], p["scene_b_date"]): p for p in body["pairs"]}
    # C pairs with B (its nearest prior scene), NOT with A as well: no combinatorial growth
    assert set(pairs) == {("2024-01-10", "2024-02-14"), ("2024-02-14", "2024-03-20")}
    ab, bc = pairs[("2024-01-10", "2024-02-14")], pairs[("2024-02-14", "2024-03-20")]
    assert ab["status"] == bc["status"] == "completed" and ab["candidates"] >= 1 and bc["candidates"] >= 1
    # the earlier pair's results survived importing the third scene, as a separate analysis
    assert body["total"] == ab["candidates"] + bc["candidates"]
    assert body["pairs"][0]["scene_b_date"] == "2024-03-20"  # newest pair first

    only_ab = client.get("/api/changes", params={"job_id": ab["job_id"]}).json()
    assert only_ab["total"] == only_ab["shown"] == ab["candidates"]
    assert {c["scene_b_date"] for c in only_ab["candidates"]} == {"2024-02-14"}
    only_bc = client.get("/api/changes", params={"job_id": bc["job_id"]}).json()
    assert {c["scene_a_date"] for c in only_bc["candidates"]} == {"2024-02-14"}
    assert len(only_ab["pairs"]) == 2  # the pair list is always the whole archive, so the chips stay visible


def test_out_of_order_import_fills_in_the_nearer_pairs(tmp_path):
    """A, then C, then B: B is between them, so it pairs with A (prior) and C (later)."""
    client = TestClient(app)
    for name, scn in [
        (NAME_A, Scenario(date="2024-01-10", noise_seed=1)),
        (NAME_C, Scenario(date="2024-03-20", noise_seed=3, patches=STRENGTHS[:2])),
        (NAME_B, Scenario(date="2024-02-14", noise_seed=2, patches=STRENGTHS[:2])),
    ]:
        client.post("/api/ingest", json={"file_path": str(make_safe(tmp_path / "src", name, scn))})
    dates = {(p["scene_a_date"], p["scene_b_date"]) for p in client.get("/api/changes").json()["pairs"]}
    assert dates == {("2024-01-10", "2024-03-20"), ("2024-01-10", "2024-02-14"), ("2024-02-14", "2024-03-20")}
