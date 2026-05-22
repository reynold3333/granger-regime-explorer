"""Custom periods CRUD with JSON persistence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd


PERIODS_FILE = Path("periods.json")


def load_periods() -> list[dict]:
    """Load periods from disk. Returns empty list if file doesn't exist."""
    if not PERIODS_FILE.exists():
        return []
    try:
        return json.loads(PERIODS_FILE.read_text())
    except Exception:
        return []


def save_periods(periods: list[dict]) -> None:
    """Persist periods to JSON file."""
    PERIODS_FILE.write_text(json.dumps(periods, indent=2, default=str))


def validate_period(name: str, start: str, end: str) -> Optional[str]:
    """Return None if valid, else error message."""
    if not name or not name.strip():
        return "Name cannot be empty"
    try:
        s = pd.to_datetime(start)
        e = pd.to_datetime(end)
    except Exception:
        return "Invalid date format"
    if s >= e:
        return "Start must be before end"
    return None


def add_period(periods: list[dict], name: str, start: str, end: str) -> tuple[list[dict], Optional[str]]:
    """Add a new period. Returns (updated list, error message or None)."""
    err = validate_period(name, start, end)
    if err:
        return periods, err
    if any(p["name"] == name for p in periods):
        return periods, f"Period '{name}' already exists"
    new_period = {"name": name.strip(), "start": str(start), "end": str(end)}
    return periods + [new_period], None


def remove_period(periods: list[dict], name: str) -> list[dict]:
    return [p for p in periods if p["name"] != name]
