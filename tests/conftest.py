import pytest

from edge_iot_agent.detector import StreamingDetector
from edge_iot_agent.pipeline import detect
from edge_iot_agent.simulator import FleetConfig, plan_faults, simulate
from edge_iot_agent.store import Store


@pytest.fixture(scope="session")
def cfg():
    return FleetConfig(devices=5, hours=8, seed=7)


@pytest.fixture(scope="session")
def faults(cfg):
    return plan_faults(cfg)


@pytest.fixture(scope="session")
def run(cfg, faults):
    return detect(simulate(cfg, faults), Store(), StreamingDetector())
