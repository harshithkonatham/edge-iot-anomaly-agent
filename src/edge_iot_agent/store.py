"""DuckDB storage for raw readings, plus the queries the investigator uses as tools."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from .simulator import SENSORS, Reading

SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    ts TIMESTAMP NOT NULL,
    device_id VARCHAR NOT NULL,
    sensor VARCHAR NOT NULL,
    value DOUBLE NOT NULL
);
"""


class Store:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.con = duckdb.connect(str(path))
        self.con.execute(SCHEMA)

    def add_readings(self, readings: list[Reading]) -> None:
        if not readings:
            return
        batch = pd.DataFrame(
            {
                "ts": [r.ts for r in readings],
                "device_id": [r.device_id for r in readings],
                "sensor": [r.sensor for r in readings],
                "value": [r.value for r in readings],
            }
        )
        self.con.register("batch", batch)
        self.con.execute("INSERT INTO readings SELECT * FROM batch")
        self.con.unregister("batch")

    def devices(self) -> list[str]:
        rows = self.con.execute("SELECT DISTINCT device_id FROM readings ORDER BY 1").fetchall()
        return [r[0] for r in rows]

    def _check(self, sensor: str, device_id: str | None = None) -> None:
        if sensor not in SENSORS:
            raise ValueError(f"Unknown sensor {sensor!r}. Sensors: {list(SENSORS)}")
        if device_id is not None and device_id not in self.devices():
            raise ValueError(f"Unknown device {device_id!r}. Devices: {self.devices()}")

    def window_stats(self, device_id: str, sensor: str, start: datetime, end: datetime) -> dict:
        """Summary of one sensor on one device between start and end."""
        self._check(sensor, device_id)
        row = self.con.execute(
            """
            SELECT count(*), avg(value), min(value), max(value), stddev_samp(value),
                   regr_slope(value, epoch(ts)) * 3600,
                   count(DISTINCT value)
            FROM readings
            WHERE device_id = ? AND sensor = ? AND ts BETWEEN ? AND ?
            """,
            [device_id, sensor, start, end],
        ).fetchone()
        n, mean, lo, hi, std, slope, distinct = row
        if not n:
            return {"device_id": device_id, "sensor": sensor, "readings": 0}
        return {
            "device_id": device_id,
            "sensor": sensor,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "readings": n,
            "mean": _r(mean),
            "min": _r(lo),
            "max": _r(hi),
            "std": _r(std),
            "slope_per_hour": _r(slope),
            "distinct_values": distinct,
        }

    def fleet_comparison(self, sensor: str, start: datetime, end: datetime) -> list[dict]:
        """Mean and max of one sensor for every device in the window, highest mean first."""
        self._check(sensor)
        rows = self.con.execute(
            """
            SELECT device_id, avg(value), max(value), stddev_samp(value)
            FROM readings
            WHERE sensor = ? AND ts BETWEEN ? AND ?
            GROUP BY device_id
            ORDER BY 2 DESC
            """,
            [sensor, start, end],
        ).fetchall()
        return [{"device_id": d, "mean": _r(m), "max": _r(x), "std": _r(s)} for d, m, x, s in rows]

    def series(self, device_id: str, start: datetime, end: datetime) -> pd.DataFrame:
        return self.con.execute(
            """
            SELECT ts, sensor, value FROM readings
            WHERE device_id = ? AND ts BETWEEN ? AND ?
            ORDER BY ts
            """,
            [device_id, start, end],
        ).df()

    def count(self) -> int:
        return self.con.execute("SELECT count(*) FROM readings").fetchone()[0]


def _r(value: float | None) -> float | None:
    return None if value is None else round(float(value), 3)
