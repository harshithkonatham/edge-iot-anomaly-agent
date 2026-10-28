"""Simulated fleet of industrial motors streaming temperature, vibration and current.

Faults are injected with known start and end times so detection and diagnosis can be scored
against ground truth. Readings are yielded in time order, like messages off a queue.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

# sensor -> (normal mean, normal standard deviation)
SENSORS: dict[str, tuple[float, float]] = {
    "temperature_c": (55.0, 1.2),
    "vibration_mm_s": (2.0, 0.15),
    "current_a": (12.0, 0.35),
}
FAULT_KINDS = ("bearing_wear", "overheating", "electrical_fault", "sensor_dropout")


@dataclass(frozen=True)
class Reading:
    ts: datetime
    device_id: str
    sensor: str
    value: float


@dataclass(frozen=True)
class Fault:
    device_id: str
    kind: str
    start: datetime
    end: datetime
    sensor: str | None = None  # only for sensor_dropout


@dataclass
class FleetConfig:
    devices: int = 6
    hours: float = 12
    interval_s: int = 60
    seed: int = 7
    start: datetime = field(default_factory=lambda: datetime(2026, 1, 5, 6, 0))

    @property
    def device_ids(self) -> list[str]:
        return [f"motor-{i:02d}" for i in range(1, self.devices + 1)]

    @property
    def end(self) -> datetime:
        return self.start + timedelta(hours=self.hours)


def plan_faults(cfg: FleetConfig) -> list[Fault]:
    """One fault per kind on different motors, leaving at least one motor healthy."""
    rng = random.Random(cfg.seed)
    ids = cfg.device_ids
    n = min(len(FAULT_KINDS), max(len(ids) - 1, 0))
    total = cfg.hours * 60
    faults = []
    for device, kind in zip(rng.sample(ids, n), FAULT_KINDS[:n], strict=True):
        start = cfg.start + timedelta(minutes=total * rng.uniform(0.25, 0.5))
        end = min(start + timedelta(minutes=total * rng.uniform(0.25, 0.4)), cfg.end)
        sensor = rng.choice(list(SENSORS)) if kind == "sensor_dropout" else None
        faults.append(Fault(device, kind, start, end, sensor))
    return sorted(faults, key=lambda f: f.start)


def simulate(cfg: FleetConfig, faults: list[Fault]) -> Iterator[Reading]:
    rng = np.random.default_rng(cfg.seed)
    spike_rng = random.Random(cfg.seed + 1)
    by_device: dict[str, list[Fault]] = {}
    for fault in faults:
        by_device.setdefault(fault.device_id, []).append(fault)
    last: dict[tuple[str, str], float] = {}
    frozen: dict[tuple[str, str], float] = {}

    steps = int(cfg.hours * 3600 / cfg.interval_s)
    for i in range(steps):
        ts = cfg.start + timedelta(seconds=i * cfg.interval_s)
        daily = math.sin(2 * math.pi * i * cfg.interval_s / 86400)  # gentle day/night cycle
        for device in cfg.device_ids:
            for sensor, (mean, sd) in SENSORS.items():
                value = mean + rng.normal(0, sd)
                if sensor == "temperature_c":
                    value += daily
                for fault in by_device.get(device, []):
                    if fault.start <= ts < fault.end:
                        progress = (ts - fault.start) / (fault.end - fault.start)
                        value = _apply_fault(
                            fault, sensor, value, progress, spike_rng, last, frozen, device
                        )
                value = round(value, 3)
                last[(device, sensor)] = value
                yield Reading(ts, device, sensor, value)


def _apply_fault(fault, sensor, value, progress, spike_rng, last, frozen, device) -> float:
    if fault.kind == "bearing_wear":
        # Worn bearings: vibration climbs steadily, friction adds some heat.
        bumps = {"vibration_mm_s": 3.5 * progress, "temperature_c": 4.0 * progress}
        return value + bumps.get(sensor, 0.0)
    if fault.kind == "overheating":
        # Blocked cooling: temperature climbs, the motor draws a little more current.
        bumps = {"temperature_c": 18.0 * progress, "current_a": 2.0 * progress}
        return value + bumps.get(sensor, 0.0)
    if fault.kind == "electrical_fault" and sensor == "current_a" and spike_rng.random() < 0.25:
        # Intermittent insulation or connection fault: short current spikes.
        return value * spike_rng.uniform(1.8, 2.3)
    if fault.kind == "sensor_dropout" and sensor == fault.sensor:
        # The sensor gets stuck and keeps reporting its last value.
        key = (device, sensor)
        frozen.setdefault(key, last.get(key, value))
        return frozen[key]
    return value
