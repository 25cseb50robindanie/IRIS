"""Tests for the seasonal persistence filter: same-season history decides whether a vegetation drop is the calendar."""

import sys
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from catalog import changes as store
from catalog.database import init_connection
from change_detection import params, run_change_detection
from change_detection.seasonal import annotate_job, eligible, same_season
from main import app
from s2_factory import Scenario, ingest_without_embedding, make_safe

BOX = (200, 260, 300, 380)  # vegetation -> bare between the two dates
DATE_A, DATE_B = "2024-01-10", "2024-02-14"


def name_for(day: str) -> str:
    stamp = day.replace("-", "")
    return f"S2A_MSIL2A_{stamp}T050649_N0510_R019_T43PHM_{stamp}T090000"


def ingest(tmp_path, day: str, kind: str, seed: int) -> str:
    return ingest_without_embedding(
        make_safe(tmp_path / "src", name_for(day), Scenario(date=day, noise_seed=seed, landcover=[(*BOX, kind)]))
    )


def analysed(tmp_path, prior_days=(), prior_kind="bare"):
    """The vegetation -> bare pair, plus scenes of the same ground on `prior_days`, then the full analysis."""
    a = ingest(tmp_path, DATE_A, "vegetation", 1)
    b = ingest(tmp_path, DATE_B, "bare", 2)
    for i, day in enumerate(prior_days):
        ingest(tmp_path, day, prior_kind, 10 + i)
    run_change_detection(a, b)
    return a, b


def only_candidate():
    conn = init_connection()
    try:
        (cand,) = store.list_candidates(conn)
        return cand
    finally:
        conn.close()


def raw_confidence(c) -> float:
    w = params.CONFIDENCE_WEIGHTS
    return (
        w["alignment_quality"] * c["alignment_quality"]
        + w["cluster_distance"] * c["norm_cluster_dist"]
        + w["terrain_flatness"] * c["terrain_flatness"]
        + w["valid_coverage"] * c["valid_coverage"]
    )


SEASON_DAYS = ("2021-02-10", "2022-02-20", "2023-02-05")  # three earlier years, within 30 days of 14 February


# ---- the rules on their own --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prior,expected",
    [
        ("2023-02-14", True),  # the same day a year earlier
        ("2023-01-20", True),  # 25 days earlier in the year
        ("2023-03-15", True),  # 29 days later
        ("2023-03-20", False),  # 34 days later
        ("2022-08-14", False),  # the wrong season
        ("2024-02-01", False),  # the same year is not "a prior year"
        ("2025-02-14", False),  # a later year
    ],
)
def test_same_season_window(prior, expected):
    assert same_season(date.fromisoformat(prior), date(2024, 2, 14)) is expected


def test_same_season_wraps_around_new_year_and_leap_day():
    assert same_season(date(2022, 12, 25), date(2024, 1, 10))  # 16 days apart across the year boundary
    assert same_season(date(2020, 2, 29), date(2024, 3, 5))  # 29 February with no counterpart in 2024


def test_only_vegetation_changes_with_a_real_ndvi_move_are_eligible():
    base = {"change_type": "clearance", "mean_dndvi": -0.4}
    assert eligible(base) and eligible({**base, "change_type": "vegetation_retreat"})
    assert eligible({**base, "change_type": "vegetation_loss"})  # the name used before direction classification
    assert not eligible({**base, "change_type": "construction"})
    assert not eligible({**base, "change_type": "unclassified"})
    assert not eligible({**base, "mean_dndvi": -params.NDVI_DROP_SIG})  # must exceed the threshold, not equal it
    assert eligible({**base, "mean_dndvi": -0.16})
    assert not eligible({**base, "mean_dndvi": None})


# ---- through the pipeline ----------------------------------------------------------------------------------------


def test_without_history_a_vegetation_drop_is_unverified_and_discounted(tmp_path):
    analysed(tmp_path)
    cand = only_candidate()
    assert (cand["change_type"], cand["seasonality_status"]) == ("clearance", "unverified")
    assert cand["confidence"] == pytest.approx(raw_confidence(cand) * params.UNVERIFIED_CONFIDENCE_FACTOR, abs=1e-4)
    assert cand["direction_evidence"]["seasonality"]["priors_covering"] == 0

    conn = init_connection()
    try:
        details = store.list_jobs(conn)[0]["details"]
    finally:
        conn.close()
    assert details["seasonality"] == {"eligible": 1, "seasonal": 0, "anomalous": 0, "unverified": 1, "prior_scenes": 0}


def test_history_that_looks_like_this_season_marks_the_drop_seasonal(tmp_path):
    analysed(tmp_path, SEASON_DAYS, prior_kind="bare")  # the ground is bare every February
    cand = only_candidate()
    assert cand["seasonality_status"] == "seasonal"
    assert cand["confidence"] == pytest.approx(raw_confidence(cand) * params.SEASONAL_CONFIDENCE_FACTOR, abs=1e-4)
    ev = cand["direction_evidence"]["seasonality"]
    assert ev["priors_clear"] == 3 and abs(ev["prior_mean_ndvi"] - ev["current_ndvi"]) < 0.05
    assert ev["drop_in_std"] <= params.SEASONAL_SIGMAS


def test_a_break_in_the_history_is_anomalous_and_keeps_its_confidence(tmp_path):
    analysed(tmp_path, SEASON_DAYS, prior_kind="vegetation")  # green every February until this one
    cand = only_candidate()
    assert cand["seasonality_status"] == "anomalous"
    assert cand["confidence"] == pytest.approx(raw_confidence(cand), abs=1e-4)
    ev = cand["direction_evidence"]["seasonality"]
    assert ev["prior_mean_ndvi"] > 0.5 > ev["current_ndvi"] and ev["drop_in_std"] > params.SEASONAL_SIGMAS


def test_two_priors_are_not_enough(tmp_path):
    analysed(tmp_path, SEASON_DAYS[:2], prior_kind="bare")
    cand = only_candidate()
    assert cand["seasonality_status"] == "unverified"
    assert cand["direction_evidence"]["seasonality"]["priors_covering"] == 2


def test_scenes_from_another_season_or_year_do_not_count(tmp_path):
    wrong_season = ("2021-08-10", "2022-08-12", "2023-08-15")
    analysed(tmp_path, wrong_season + ("2024-02-01",), prior_kind="bare")  # ...and one from the same year
    cand = only_candidate()
    assert cand["seasonality_status"] == "unverified" and cand["direction_evidence"]["seasonality"]["priors_covering"] == 0


def test_a_change_that_is_not_vegetation_is_not_checked(tmp_path):
    a = ingest_without_embedding(make_safe(tmp_path / "src", name_for(DATE_A), Scenario(date=DATE_A, noise_seed=1)))
    b = ingest_without_embedding(
        make_safe(tmp_path / "src", name_for(DATE_B), Scenario(date=DATE_B, noise_seed=2, patches=[(*BOX, 0.5)]))  # NIR-only change: unclassified
    )
    run_change_detection(a, b)
    conn = init_connection()
    try:
        cands = store.list_candidates(conn)
    finally:
        conn.close()
    assert cands and all(c["change_type"] not in params.SEASONAL_TYPES for c in cands)
    assert all(c["seasonality_status"] is None for c in cands)  # the check did not apply: no badge, no discount


# ---- through the API ---------------------------------------------------------------------------------------------


def test_the_status_reaches_the_list_the_detail_the_trace_and_the_export(tmp_path):
    analysed(tmp_path, SEASON_DAYS, prior_kind="bare")
    client = TestClient(app)
    listed = client.get("/api/changes").json()["candidates"][0]
    assert listed["seasonality_status"] == "seasonal"

    detail = client.get(f"/api/changes/{listed['candidate_id']}").json()
    assert detail["seasonality_status"] == "seasonal" and detail["seasonality"]["priors_clear"] == 3
    assert detail["confidence_factor"] == params.SEASONAL_CONFIDENCE_FACTOR
    lines = {line["key"]: line["text"] for line in detail["processing_details"]}
    assert lines["seasonality"].startswith("Seasonal — NDVI") and "over 3 prior years" in lines["seasonality"]

    exported = client.get("/api/export/changes").json()["features"][0]["properties"]
    assert exported["seasonality_status"] == "seasonal"


def test_hide_seasonal_leaves_out_seasonal_candidates_only(tmp_path):
    analysed(tmp_path, SEASON_DAYS, prior_kind="bare")
    client = TestClient(app)
    assert client.get("/api/changes").json()["shown"] == 1  # off by default at the API
    hidden = client.get("/api/changes", params={"hide_seasonal": "true"}).json()
    assert hidden["shown"] == 0 and hidden["candidates"] == [] and hidden["total"] == 1  # stored, just not listed


def test_hide_seasonal_keeps_unverified_and_anomalous(tmp_path):
    analysed(tmp_path)  # unverified
    client = TestClient(app)
    assert client.get("/api/changes", params={"hide_seasonal": "true"}).json()["shown"] == 1


def test_the_check_is_stored_on_the_row_for_the_search_results(tmp_path):
    analysed(tmp_path)
    conn = init_connection()
    try:
        rows = store.query_candidates(conn, hide_seasonal=True)
    finally:
        conn.close()
    assert [r["seasonality_status"] for r in rows] == ["unverified"]


# ---- applying it to a catalog analysed before the filter existed --------------------------------------------------


def test_annotate_job_checks_stored_candidates_once(tmp_path):
    analysed(tmp_path, SEASON_DAYS, prior_kind="bare")
    cand = only_candidate()
    raw = raw_confidence(cand)

    # Put the row back as it was before the filter existed
    conn = init_connection()
    try:
        conn.execute("UPDATE change_candidates SET seasonality_status = NULL, confidence = ?", (raw,))
        conn.commit()
    finally:
        conn.close()

    trace = annotate_job(cand["job_id"])
    assert (trace["eligible"], trace["seasonal"]) == (1, 1)
    after = only_candidate()
    assert after["seasonality_status"] == "seasonal"
    assert after["confidence"] == pytest.approx(raw * params.SEASONAL_CONFIDENCE_FACTOR, abs=1e-4)

    again = annotate_job(cand["job_id"])  # already checked: nothing to do, so the confidence is not halved twice
    assert again["eligible"] == 0
    assert only_candidate()["confidence"] == pytest.approx(after["confidence"], abs=1e-9)

    with pytest.raises(ValueError):
        annotate_job(9999)
