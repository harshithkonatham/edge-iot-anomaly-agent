"""The structured diagnosis both the Claude investigator and the offline rules produce."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Cause = Literal["bearing_wear", "overheating", "electrical_fault", "sensor_fault", "unknown"]

# Ground-truth fault kind -> the diagnosis we expect for it.
EXPECTED_CAUSE: dict[str, Cause] = {
    "bearing_wear": "bearing_wear",
    "overheating": "overheating",
    "electrical_fault": "electrical_fault",
    "sensor_dropout": "sensor_fault",
}


class Diagnosis(BaseModel):
    likely_cause: Cause
    confidence: Literal["low", "medium", "high"]
    severity: Literal["low", "medium", "high", "critical"]
    summary: str = Field(description="Two or three sentences a maintenance engineer can act on.")
    evidence: list[str] = Field(description="The specific readings or comparisons relied on.")
    recommended_actions: list[str] = Field(description="Ordered steps, from the runbook.")
