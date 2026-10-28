"""Stream readings through the detector into DuckDB, then diagnose and score each incident."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import timedelta

from .detector import Anomaly, Incident, StreamingDetector, group_incidents
from .models import EXPECTED_CAUSE, Diagnosis
from .simulator import Fault, Reading
from .store import Store

BATCH_SIZE = 5000
MATCH_GRACE = timedelta(minutes=30)  # an incident may end a little after the fault does


@dataclass
class DetectionRun:
    store: Store
    incidents: list[Incident]
    baselines: dict
    readings: int
    anomalies: int


def detect(stream: Iterable[Reading], store: Store, detector: StreamingDetector) -> DetectionRun:
    """Consume a stream the way an edge gateway would: score each reading, buffer, flush."""
    buffer: list[Reading] = []
    anomalies: list[Anomaly] = []
    count = 0
    for reading in stream:
        count += 1
        buffer.append(reading)
        if (anomaly := detector.process(reading)) is not None:
            anomalies.append(anomaly)
        if len(buffer) >= BATCH_SIZE:
            store.add_readings(buffer)
            buffer.clear()
    store.add_readings(buffer)
    return DetectionRun(
        store=store,
        incidents=group_incidents(anomalies),
        baselines=detector.baselines(),
        readings=count,
        anomalies=len(anomalies),
    )


def diagnose_all(
    incidents: list[Incident], diagnose: Callable[[Incident], Diagnosis]
) -> dict[str, Diagnosis]:
    return {incident.id: diagnose(incident) for incident in incidents}


@dataclass
class Score:
    faults: int
    incidents: int
    detected_faults: int
    false_alarms: int
    correct_causes: int
    diagnosed: int
    delays_min: list[float] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return (self.incidents - self.false_alarms) / self.incidents if self.incidents else 1.0

    @property
    def recall(self) -> float:
        return self.detected_faults / self.faults if self.faults else 1.0

    @property
    def diagnosis_accuracy(self) -> float:
        return self.correct_causes / self.diagnosed if self.diagnosed else 0.0

    @property
    def mean_delay_min(self) -> float | None:
        return sum(self.delays_min) / len(self.delays_min) if self.delays_min else None


def _matches(incident: Incident, fault: Fault) -> bool:
    return (
        incident.device_id == fault.device_id
        and incident.start <= fault.end + MATCH_GRACE
        and incident.end >= fault.start
    )


def score(faults: list[Fault], incidents: list[Incident], diagnoses: dict[str, Diagnosis]) -> Score:
    """Compare detected incidents and diagnoses with the injected ground truth."""
    result = Score(len(faults), len(incidents), 0, 0, 0, 0)
    matched_incidents: set[str] = set()
    for fault in faults:
        hits = [i for i in incidents if _matches(i, fault)]
        matched_incidents.update(i.id for i in hits)
        expected = EXPECTED_CAUSE[fault.kind]
        row = {
            "device": fault.device_id,
            "fault": fault.kind,
            "expected": expected,
            "incident": None,
            "delay_min": None,
            "diagnosed": None,
        }
        if hits:
            first = min(hits, key=lambda i: i.start)
            delay = (first.start - fault.start).total_seconds() / 60
            result.detected_faults += 1
            result.delays_min.append(delay)
            row.update(incident=first.id, delay_min=round(delay))
            if first.id in diagnoses:
                cause = diagnoses[first.id].likely_cause
                result.diagnosed += 1
                result.correct_causes += cause == expected
                row["diagnosed"] = cause
        result.rows.append(row)
    result.false_alarms = sum(1 for i in incidents if i.id not in matched_incidents)
    return result
