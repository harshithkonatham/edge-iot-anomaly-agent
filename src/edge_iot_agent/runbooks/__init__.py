"""Maintenance runbooks shipped with the package, one Markdown file per fault type."""

from __future__ import annotations

import re
from importlib import resources

TOPICS = ("bearing_wear", "overheating", "electrical_fault", "sensor_fault")


def read(topic: str) -> str:
    if topic not in TOPICS:
        raise ValueError(f"Unknown runbook {topic!r}. Runbooks: {list(TOPICS)}")
    return resources.files(__package__).joinpath(f"{topic}.md").read_text(encoding="utf-8")


def actions_for(topic: str) -> list[str]:
    """The numbered action steps of a runbook."""
    if topic not in TOPICS:
        return ["Inspect the motor and review the raw sensor data with an engineer."]
    return re.findall(r"^\d+\.\s+(.*)$", read(topic), flags=re.MULTILINE)
