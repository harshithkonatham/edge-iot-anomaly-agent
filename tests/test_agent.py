"""Investigator wiring tests with a fake Anthropic client: no network, no API key."""

import json
from types import SimpleNamespace

import pytest

from edge_iot_agent.agent import InvestigationRefused, Investigator, build_tools
from edge_iot_agent.models import Diagnosis


class FakeRunner:
    """Plays Claude: calls the real tools, then returns a diagnosis."""

    def __init__(self, tools, outcome, incident_id):
        self.tools = {t.name: t for t in tools}
        self.incident_id = incident_id
        self.outcome = outcome
        self.seen = {}

    def until_done(self):
        incident = json.loads(self.tools["get_incident"].call({"incident_id": self.incident_id}))
        device, sensor = incident["device_id"], next(iter(incident["sensors"]))
        self.seen["window"] = json.loads(
            self.tools["sensor_window"].call(
                {
                    "device_id": device,
                    "sensor": sensor,
                    "start": incident["start"],
                    "end": incident["end"],
                }
            )
        )
        self.seen["baseline"] = json.loads(
            self.tools["sensor_baseline"].call({"device_id": device, "sensor": sensor})
        )
        self.seen["fleet"] = json.loads(
            self.tools["compare_with_fleet"].call(
                {"sensor": sensor, "start": incident["start"], "end": incident["end"]}
            )
        )
        self.seen["runbook"] = json.loads(
            self.tools["read_runbook"].call({"topic": "sensor_fault"})
        )
        if self.outcome == "refusal":
            return SimpleNamespace(stop_reason="refusal", stop_details=None, parsed_output=None)
        if self.outcome == "empty":
            return SimpleNamespace(stop_reason="max_tokens", parsed_output=None)
        diagnosis = Diagnosis(
            likely_cause="sensor_fault",
            confidence="high",
            severity="high",
            summary="Temperature sensor is stuck.",
            evidence=[f"distinct values: {self.seen['window']['distinct_values']}"],
            recommended_actions=["Replace the sensor."],
        )
        return SimpleNamespace(stop_reason="end_turn", parsed_output=diagnosis)


class FakeClient:
    def __init__(self, outcome="ok"):
        self.kwargs = None
        self.runner = None
        self.outcome = outcome
        self.beta = SimpleNamespace(messages=SimpleNamespace(tool_runner=self._tool_runner))

    def _tool_runner(self, **kwargs):
        self.kwargs = kwargs
        incident_id = kwargs["messages"][0]["content"].split()[2].rstrip(":")
        self.runner = FakeRunner(kwargs["tools"], self.outcome, incident_id)
        return self.runner


def _investigate(run, outcome="ok"):
    client = FakeClient(outcome)
    investigator = Investigator(run.store, run.baselines, client=client)
    by_id = {i.id: i for i in run.incidents}
    stuck = next(i for i in run.incidents if any("flatline" in s.kinds for s in i.sensors.values()))
    return client, investigator.diagnose(stuck, by_id)


def test_investigator_uses_tools_and_returns_diagnosis(run):
    client, diagnosis = _investigate(run)
    assert diagnosis.likely_cause == "sensor_fault"
    assert client.kwargs["model"] == "claude-opus-5-5"
    assert client.kwargs["output_format"] is Diagnosis
    assert client.kwargs["fallbacks"] == "default"
    assert client.kwargs["output_config"] == {"effort": "medium"}
    seen = client.runner.seen
    assert seen["window"]["distinct_values"] == 1  # the stuck sensor repeats one value
    assert "median" in seen["baseline"]
    assert len(seen["fleet"]) == 5
    assert "Runbook: sensor fault" in seen["runbook"]["runbook"]


@pytest.mark.parametrize("outcome", ["refusal", "empty"])
def test_investigator_raises_without_a_diagnosis(run, outcome):
    with pytest.raises(InvestigationRefused):
        _investigate(run, outcome)


def test_tools_return_errors_to_claude(run):
    tools = {t.name: t for t in build_tools(run.store, {}, run.baselines)}
    assert tools["get_incident"].call({"incident_id": "INC-999"}).startswith("Error:")
    assert (
        tools["sensor_baseline"]
        .call({"device_id": "motor-99", "sensor": "current_a"})
        .startswith("Error:")
    )
    bad_time = tools["compare_with_fleet"].call(
        {"sensor": "current_a", "start": "yesterday", "end": "today"}
    )
    assert bad_time.startswith("Error:")
    assert tools["read_runbook"].call({"topic": "nope"}).startswith("Error:")
