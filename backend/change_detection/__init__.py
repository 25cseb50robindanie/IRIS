"""IRIS Change Detection — Phases 1-5 of the false-alarm suppression pipeline."""

from .pipeline import run_change_detection
from .trigger import schedule_change_detection

__all__ = ["run_change_detection", "schedule_change_detection"]
