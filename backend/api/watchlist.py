"""IRIS Watchlist API Router — locations the analyst wants watched, and the alerts raised when a new detection lands on one.

"The system watches while you don't": the check is made by the backend every time a change-detection job completes
(catalog/watchlist.py), not by the page that happens to be open.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field, model_validator

from catalog import watchlist as store
from catalog.database import init_connection, init_schema

logger = logging.getLogger("iris.api.watchlist")
router = APIRouter(prefix="/api", tags=["watchlist"])

DEFAULT_RADIUS_M = 500.0
MAX_RADIUS_M = 50_000.0


class WatchRequest(BaseModel):
    """A location to watch: a box (WGS84 [min_lon, min_lat, max_lon, max_lat]), or a centre point [lon, lat] and a radius."""
    name: Optional[str] = Field(default=None, max_length=100)
    bounds: Optional[List[float]] = Field(default=None, min_length=4, max_length=4)
    center: Optional[List[float]] = Field(default=None, min_length=2, max_length=2)
    radius_m: float = Field(default=DEFAULT_RADIUS_M, gt=0, le=MAX_RADIUS_M)
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _one_place(self) -> "WatchRequest":
        if (self.bounds is None) == (self.center is None):
            raise ValueError("Give exactly one of bounds or center")
        if self.bounds is not None:
            min_lon, min_lat, max_lon, max_lat = self.bounds
            if not (-180 <= min_lon < max_lon <= 180 and -90 <= min_lat < max_lat <= 90):
                raise ValueError("bounds must be [min_lon, min_lat, max_lon, max_lat] within the world, with min below max")
        else:
            lon, lat = self.center  # type: ignore[misc]
            if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                raise ValueError("center must be [lon, lat] within the world")
        return self


class WatchedLocation(BaseModel):
    id: int
    name: str
    bounds: List[float]  # [min_lon, min_lat, max_lon, max_lat]
    confidence_threshold: float
    created_at: str
    alerts: int  # alerts ever raised here
    unacknowledged_alerts: int


class WatchAlert(BaseModel):
    """A detection that landed on a watched location and reached its threshold."""
    alert_id: int
    watchlist_id: int
    watch_name: str
    candidate_id: int
    job_id: int
    confidence: float
    created_at: str
    acknowledged_at: Optional[str] = None
    bounds: List[float]  # the detection's box, [min_lon, min_lat, max_lon, max_lat]
    change_type: Optional[str] = None
    direction: Optional[str] = None
    area_px: Optional[int] = None
    area_ha: Optional[float] = None
    mgrs: Optional[str] = None
    scene_a_date: Optional[str] = None
    scene_b_date: Optional[str] = None
    review_status: str


class AlertList(BaseModel):
    total: int
    unacknowledged: int
    alerts: List[WatchAlert]


@router.post("/watchlist", response_model=WatchedLocation, status_code=status.HTTP_201_CREATED)
def add_watch(payload: WatchRequest) -> WatchedLocation:
    """Watch a location. Detections from now on that overlap it and reach `confidence_threshold` raise an alert."""
    bounds = payload.bounds or store.box_around(payload.center[0], payload.center[1], payload.radius_m)  # type: ignore[index]
    conn = init_connection()
    try:
        init_schema(conn)
        try:
            return WatchedLocation(**store.add_location(conn, payload.name, bounds, payload.confidence_threshold))
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    finally:
        conn.close()


@router.get("/watchlist", response_model=List[WatchedLocation])
def list_watch() -> List[WatchedLocation]:
    conn = init_connection()
    try:
        init_schema(conn)
        return [WatchedLocation(**w) for w in store.list_locations(conn)]
    finally:
        conn.close()


# The alert routes come before /watchlist/{id}: "alerts" must not be read as an id
@router.get("/watchlist/alerts", response_model=AlertList)
def list_alerts(unacknowledged_only: bool = Query(default=False)) -> AlertList:
    """Candidates that triggered watchlist hits, newest first."""
    conn = init_connection()
    try:
        init_schema(conn)
        every = store.list_alerts(conn)
        shown = [a for a in every if not (unacknowledged_only and a["acknowledged_at"])]
    finally:
        conn.close()
    return AlertList(
        total=len(every),
        unacknowledged=sum(1 for a in every if not a["acknowledged_at"]),
        alerts=[
            WatchAlert(bounds=[a["min_lon"], a["min_lat"], a["max_lon"], a["max_lat"]], **{k: v for k, v in a.items() if k not in ("min_lon", "min_lat", "max_lon", "max_lat")})
            for a in shown
        ],
    )


@router.post("/watchlist/alerts/acknowledge")
def acknowledge_all() -> dict:
    """The analyst has seen every open alert."""
    conn = init_connection()
    try:
        init_schema(conn)
        return {"acknowledged": store.acknowledge(conn)}
    finally:
        conn.close()


@router.post("/watchlist/alerts/{alert_id}/acknowledge")
def acknowledge_one(alert_id: int) -> dict:
    conn = init_connection()
    try:
        init_schema(conn)
        if not any(a["alert_id"] == alert_id for a in store.list_alerts(conn)):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown alert: {alert_id}")
        return {"acknowledged": store.acknowledge(conn, alert_id)}
    finally:
        conn.close()


@router.delete("/watchlist/{watch_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_watch(watch_id: int) -> None:
    conn = init_connection()
    try:
        init_schema(conn)
        if not store.delete_location(conn, watch_id):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown watched location: {watch_id}")
    finally:
        conn.close()
