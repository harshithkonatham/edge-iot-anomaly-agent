"""Streaming anomaly detection that is cheap enough to run on an edge gateway.

Per device and sensor it learns a robust baseline (median and MAD) from the first readings,
then scores a smoothed value (EWMA) against it. A sensor that repeats the exact same value is
flagged as flatlined. Nearby anomalies on the same device are grouped into incidents.
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .simulator import Reading

Key = tuple[str, str]


@dataclass(frozen=True)
class Anomaly:
    ts: datetime
    device_id: str
    sensor: str
    value: float
    score: float  # robust z-score of the smoothed value; 0 for flatlines
    kind: str  # "high", "low" or "flatline"


@dataclass
class SensorSummary:
    count: int = 0
    peak_score: float = 0.0
    peak_value: float | None = None
    kinds: set[str] = field(default_factory=set)


@dataclass
class Incident:
    id: str
    device_id: str
    start: datetime
    end: datetime
    sensors: dict[str, SensorSummary]

    @property
    def anomaly_count(self) -> int:
        return sum(s.count for s in self.sensors.values())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "device_id": self.device_id,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "minutes": round((self.end - self.start).total_seconds() / 60),
            "sensors": {
                name: {
                    "anomalies": s.count,
                    "peak_score": round(s.peak_score, 2),
                    "peak_value": s.peak_value,
                    "kinds": sorted(s.kinds),
                }
                for name, s in self.sensors.items()
            },
        }


class StreamingDetector:
    def __init__(
        self,
        warmup: int = 60,
        threshold: float = 4.0,
        alpha: float = 0.3,
        flatline_window: int = 15,
    ) -> None:
        self.warmup = warmup
        self.threshold = threshold
        self.alpha = alpha
        self.flatline_window = flatline_window
        self._warm: dict[Key, list[float]] = {}
        self._baseline: dict[Key, tuple[float, float]] = {}
        self._ewma: dict[Key, float] = {}
        self._recent: dict[Key, deque[float]] = {}

    def baseline(self, device_id: str, sensor: str) -> dict | None:
        found = self._baseline.get((device_id, sensor))
        return None if found is None else {"median": found[0], "scale": round(found[1], 4)}

    def baselines(self) -> dict[Key, dict]:
        return {key: self.baseline(*key) for key in self._baseline}

    def process(self, r: Reading) -> Anomaly | None:
        key = (r.device_id, r.sensor)
        recent = self._recent.setdefault(key, deque(maxlen=self.flatline_window))
        recent.append(r.value)

        if key not in self._baseline:
            warm = self._warm.setdefault(key, [])
            warm.append(r.value)
            if len(warm) >= self.warmup:
                median = statistics.median(warm)
                mad = statistics.median(abs(x - median) for x in warm)
                self._baseline[key] = (median, max(1.4826 * mad, 1e-6))
                self._ewma[key] = median
                del self._warm[key]
            return None

        if len(recent) == self.flatline_window and max(recent) == min(recent):
            return Anomaly(r.ts, r.device_id, r.sensor, r.value, 0.0, "flatline")

        median, scale = self._baseline[key]
        self._ewma[key] = self.alpha * r.value + (1 - self.alpha) * self._ewma[key]
        score = (self._ewma[key] - median) / scale
        if abs(score) >= self.threshold:
            kind = "high" if score > 0 else "low"
            return Anomaly(r.ts, r.device_id, r.sensor, r.value, round(score, 2), kind)
        return None


def group_incidents(
    anomalies: list[Anomaly],
    gap: timedelta = timedelta(minutes=30),
    min_anomalies: int = 3,
) -> list[Incident]:
    """Merge anomalies on the same device that are less than `gap` apart into incidents."""
    groups: list[list[Anomaly]] = []
    for a in sorted(anomalies, key=lambda a: (a.device_id, a.ts)):
        current = groups[-1] if groups else None
        if current and current[-1].device_id == a.device_id and a.ts - current[-1].ts <= gap:
            current.append(a)
        else:
            groups.append([a])

    incidents = []
    for group in groups:
        if len(group) < min_anomalies:
            continue  # isolated blips are noise, not incidents
        sensors: dict[str, SensorSummary] = {}
        for a in group:
            s = sensors.setdefault(a.sensor, SensorSummary())
            s.count += 1
            s.kinds.add(a.kind)
            if abs(a.score) >= abs(s.peak_score):
                s.peak_score, s.peak_value = a.score, a.value
        incidents.append(Incident("", group[0].device_id, group[0].ts, group[-1].ts, sensors))

    incidents.sort(key=lambda i: i.start)
    for n, incident in enumerate(incidents, 1):
        incident.id = f"INC-{n:03d}"
    return incidents
