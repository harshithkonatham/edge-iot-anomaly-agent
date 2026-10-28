"""Claude incident investigator.

For each incident the detector raises, Claude investigates with read-only tools over the
DuckDB store (sensor windows, baselines, fleet comparison) and the maintenance runbooks, then
returns a validated `Diagnosis` (structured output) a maintenance engineer can act on.
"""

from __future__ import annotations

import json
from datetime import datetime

import anthropic
from anthropic import beta_tool

from . import runbooks
from .detector import Incident
from .models import Diagnosis
from .store import Store

MODEL = "claude-opus-5-5"
# If a safety classifier declines a request, the API re-runs it on a recommended fallback model.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM = """You are a reliability engineer investigating anomalies on a fleet of industrial
electric motors. Each motor reports temperature_c, vibration_mm_s and current_a every minute.

For the incident you are given, use the tools to find out what is wrong: compare the readings
with the motor's own baseline and with the rest of the fleet, look at trends, and read the
runbook for the cause you suspect. A stuck sensor is a sensor problem, not a motor problem;
a fleet-wide change points to the environment rather than one motor.

Return the diagnosis with recommended actions taken from the runbook. Put the specific
numbers you relied on in the evidence list. If the data doesn't support a confident cause,
say "unknown" with low confidence rather than guessing."""


class InvestigationRefused(RuntimeError):
    """Claude declined the request even after the server-side fallback."""


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def build_tools(store: Store, incidents: dict[str, Incident], baselines: dict) -> list:
    def run(fn) -> str:
        try:
            return json.dumps(fn(), default=str)
        except ValueError as exc:  # bad ids or timestamps go back to Claude to fix
            return f"Error: {exc}"

    @beta_tool
    def get_incident(incident_id: str) -> str:
        """Details of an incident: device, time window and per-sensor anomaly summary.

        Args:
            incident_id: Incident id such as INC-003.
        """

        def fetch():
            if incident_id not in incidents:
                raise ValueError(f"Unknown incident {incident_id!r}")
            return incidents[incident_id].to_dict()

        return run(fetch)

    @beta_tool
    def sensor_baseline(device_id: str, sensor: str) -> str:
        """The normal level for a sensor on a motor (median and robust spread), learned
        from its first hour of data.

        Args:
            device_id: Motor id such as motor-03.
            sensor: temperature_c, vibration_mm_s or current_a.
        """

        def fetch():
            found = baselines.get((device_id, sensor))
            if found is None:
                raise ValueError(f"No baseline for {device_id}/{sensor}")
            return {"device_id": device_id, "sensor": sensor, **found}

        return run(fetch)

    @beta_tool
    def sensor_window(device_id: str, sensor: str, start: str, end: str) -> str:
        """Stats for one sensor on one motor in a time window: mean, min, max, std,
        slope per hour and number of distinct values (1 means the sensor is stuck).

        Args:
            device_id: Motor id such as motor-03.
            sensor: temperature_c, vibration_mm_s or current_a.
            start: ISO timestamp, e.g. 2026-01-05T09:30:00.
            end: ISO timestamp.
        """
        return run(lambda: store.window_stats(device_id, sensor, _time(start), _time(end)))

    @beta_tool
    def compare_with_fleet(sensor: str, start: str, end: str) -> str:
        """Mean, max and std of one sensor for every motor in a time window, highest first.

        Args:
            sensor: temperature_c, vibration_mm_s or current_a.
            start: ISO timestamp.
            end: ISO timestamp.
        """
        return run(lambda: store.fleet_comparison(sensor, _time(start), _time(end)))

    @beta_tool
    def read_runbook(topic: str) -> str:
        """Maintenance runbook: fault signature, severity thresholds and actions.

        Args:
            topic: bearing_wear, overheating, electrical_fault or sensor_fault.
        """
        return run(lambda: {"topic": topic, "runbook": runbooks.read(topic)})

    return [get_incident, sensor_baseline, sensor_window, compare_with_fleet, read_runbook]


class Investigator:
    def __init__(
        self,
        store: Store,
        baselines: dict,
        client: anthropic.Anthropic | None = None,
        model: str = MODEL,
        effort: str = "medium",
        max_iterations: int = 12,
    ) -> None:
        self.store = store
        self.baselines = baselines
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.max_iterations = max_iterations

    def diagnose(self, incident: Incident, incidents: dict[str, Incident]) -> Diagnosis:
        runner = self.client.beta.messages.tool_runner(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM,
            tools=build_tools(self.store, incidents, self.baselines),
            messages=[
                {
                    "role": "user",
                    "content": f"Investigate incident {incident.id}:\n"
                    f"{json.dumps(incident.to_dict())}",
                }
            ],
            output_format=Diagnosis,
            output_config={"effort": self.effort},
            betas=[FALLBACK_BETA],
            fallbacks="default",
            max_iterations=self.max_iterations,
        )
        final = runner.until_done()
        if final.stop_reason == "refusal":
            raise InvestigationRefused(f"Claude declined: {getattr(final, 'stop_details', None)}")
        if final.parsed_output is None:
            raise InvestigationRefused(f"No diagnosis returned (stop: {final.stop_reason})")
        return final.parsed_output
