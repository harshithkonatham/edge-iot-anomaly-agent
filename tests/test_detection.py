from datetime import datetime, timedelta

import pytest

from edge_iot_agent import runbooks
from edge_iot_agent.detector import Anomaly, StreamingDetector, group_incidents
from edge_iot_agent.simulator import SENSORS, FleetConfig, Reading, plan_faults, simulate

T0 = datetime(2026, 1, 1)


def _reading(i, value, sensor="current_a"):
    return Reading(T0 + timedelta(minutes=i), "motor-01", sensor, value)


def test_fault_plan_leaves_a_healthy_motor_and_is_deterministic(cfg, faults):
    assert len(faults) == 4
    assert len({f.device_id for f in faults}) == 4
    assert faults == plan_faults(cfg)
    assert all(cfg.start < f.start < f.end <= cfg.end for f in faults)


def test_simulation_volume_and_stuck_sensor(cfg, faults):
    readings = list(simulate(cfg, faults))
    assert len(readings) == cfg.devices * len(SENSORS) * int(cfg.hours * 60)
    dropout = next(f for f in faults if f.kind == "sensor_dropout")
    stuck = {
        r.value
        for r in readings
        if r.device_id == dropout.device_id
        and r.sensor == dropout.sensor
        and dropout.start <= r.ts < dropout.end
    }
    assert len(stuck) == 1


def test_small_fleet_still_gets_faults():
    assert len(plan_faults(FleetConfig(devices=2, hours=4))) == 1


def test_detector_warmup_then_spike():
    det = StreamingDetector(warmup=10, threshold=4)
    values = [12.0, 12.2, 11.9, 12.1, 11.8, 12.3, 12.0, 11.9, 12.2, 12.1]
    assert all(det.process(_reading(i, v)) is None for i, v in enumerate(values))
    assert det.baseline("motor-01", "current_a")["median"] == pytest.approx(12.05)
    assert det.process(_reading(11, 12.1)) is None
    anomaly = det.process(_reading(12, 30.0))
    assert anomaly.kind == "high" and anomaly.score > 4


def test_detector_flags_low_values_and_flatlines():
    det = StreamingDetector(warmup=5, threshold=4, flatline_window=4)
    for i, v in enumerate([2.0, 2.1, 1.9, 2.05, 1.95]):
        det.process(_reading(i, v, "vibration_mm_s"))
    assert det.process(_reading(6, -5.0, "vibration_mm_s")).kind == "low"
    kinds = [det.process(_reading(10 + i, 2.0, "vibration_mm_s")) for i in range(4)]
    assert kinds[-1].kind == "flatline"
    assert det.baselines()[("motor-01", "vibration_mm_s")]["scale"] > 0


def test_grouping_merges_nearby_and_drops_blips():
    def a(minute, device="motor-01", sensor="current_a", score=5.0):
        return Anomaly(T0 + timedelta(minutes=minute), device, sensor, 20.0, score, "high")

    anomalies = [a(0), a(5), a(10, sensor="temperature_c", score=9), a(200), a(3, "motor-02")]
    incidents = group_incidents(anomalies, gap=timedelta(minutes=30), min_anomalies=2)
    assert len(incidents) == 1
    inc = incidents[0]
    assert inc.id == "INC-001" and inc.anomaly_count == 3
    assert inc.to_dict()["sensors"]["temperature_c"]["peak_score"] == 9


def test_pipeline_finds_every_fault(run, faults):
    assert run.readings > 0 and run.store.count() == run.readings
    assert {i.device_id for i in run.incidents} == {f.device_id for f in faults}


def test_runbooks():
    assert "bearing" in runbooks.read("bearing_wear").lower()
    assert len(runbooks.actions_for("overheating")) == 5
    assert runbooks.actions_for("unknown")[0].startswith("Inspect")
    with pytest.raises(ValueError):
        runbooks.read("../secrets")
