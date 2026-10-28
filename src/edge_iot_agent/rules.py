"""Rule-based diagnosis: no API key needed, and a baseline to score the agent against."""

from __future__ import annotations

from .detector import Incident
from .models import Diagnosis
from .runbooks import actions_for
from .store import Store


def _vibration_severity(mm_s: float) -> str:
    """ISO 10816-style zones for small motors (see the bearing_wear runbook)."""
    for limit, severity in ((7.1, "critical"), (4.5, "high"), (2.8, "medium")):
        if mm_s >= limit:
            return severity
    return "low"


def _dominant(incident: Incident) -> list[tuple[str, float]]:
    return sorted(
        ((name, abs(s.peak_score)) for name, s in incident.sensors.items()),
        key=lambda kv: kv[1],
        reverse=True,
    )


def diagnose(incident: Incident, store: Store) -> Diagnosis:
    sensors = incident.sensors
    flat = [name for name, s in sensors.items() if "flatline" in s.kinds]
    ranked = _dominant(incident)
    top, top_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    confidence = "high" if top_score >= 2 * runner_up else "medium"

    if flat:
        cause, severity, confidence = "sensor_fault", "medium", "high"
        if "temperature_c" in flat:
            severity = "high"  # the sensor that protects the motor is blind
        focus = flat[0]
    elif top == "vibration_mm_s":
        cause, focus = "bearing_wear", top
        severity = _vibration_severity(sensors[top].peak_value or 0)
    elif top == "temperature_c":
        cause, focus = "overheating", top
        peak = sensors[top].peak_value or 0
        severity = "critical" if peak >= 80 else "high" if peak >= 70 else "medium"
    elif top == "current_a":
        cause, focus = "electrical_fault", top
        severity = "high" if sensors[top].count >= 20 else "medium"
    else:  # pragma: no cover - every sensor is handled above
        cause, focus, severity, confidence = "unknown", top, "low", "low"

    stats = store.window_stats(incident.device_id, focus, incident.start, incident.end)
    evidence = [
        f"{name}: {s.count} anomalies, peak score {s.peak_score:.1f}, "
        f"peak value {s.peak_value}, kinds {sorted(s.kinds)}"
        for name, s in sensors.items()
    ]
    evidence.append(
        f"{focus} during incident: mean {stats.get('mean')}, max {stats.get('max')}, "
        f"slope {stats.get('slope_per_hour')}/h, {stats.get('distinct_values')} distinct values"
    )
    summary = (
        f"{incident.device_id}: {cause.replace('_', ' ')} suspected from {focus} "
        f"over {round((incident.end - incident.start).total_seconds() / 60)} minutes."
    )
    return Diagnosis(
        likely_cause=cause,
        confidence=confidence,
        severity=severity,
        summary=summary,
        evidence=evidence,
        recommended_actions=actions_for(cause),
    )
