from datetime import timedelta

import pytest

from edge_iot_agent import rules
from edge_iot_agent.cli import main
from edge_iot_agent.models import Diagnosis
from edge_iot_agent.pipeline import diagnose_all, score


def test_window_stats_and_fleet(run, cfg):
    stats = run.store.window_stats("motor-01", "current_a", cfg.start, cfg.end)
    assert stats["readings"] == int(cfg.hours * 60)
    assert 10 < stats["mean"] < 14
    fleet = run.store.fleet_comparison("temperature_c", cfg.start, cfg.end)
    assert len(fleet) == cfg.devices
    assert fleet[0]["mean"] >= fleet[-1]["mean"]
    empty = run.store.window_stats(
        "motor-01", "current_a", cfg.end + timedelta(days=1), cfg.end + timedelta(days=2)
    )
    assert empty["readings"] == 0


def test_store_rejects_unknown_names(run, cfg):
    with pytest.raises(ValueError, match="Unknown sensor"):
        run.store.window_stats("motor-01", "pressure", cfg.start, cfg.end)
    with pytest.raises(ValueError, match="Unknown device"):
        run.store.window_stats("motor-99", "current_a", cfg.start, cfg.end)


def test_rules_diagnose_every_fault_correctly(run, faults):
    diagnoses = diagnose_all(run.incidents, lambda i: rules.diagnose(i, run.store))
    result = score(faults, run.incidents, diagnoses)
    assert result.recall == 1.0
    assert result.precision == 1.0
    assert result.diagnosis_accuracy == 1.0
    assert result.mean_delay_min < 90
    assert all(d.recommended_actions for d in diagnoses.values())


def test_score_counts_misses_and_false_alarms(run, faults):
    wrong = Diagnosis(
        likely_cause="unknown",
        confidence="low",
        severity="low",
        summary="?",
        evidence=[],
        recommended_actions=[],
    )
    first = run.incidents[0]
    result = score(faults, [first], {first.id: wrong})
    assert result.detected_faults == 1 and result.recall == 0.25
    assert result.diagnosis_accuracy == 0.0
    shifted = score([], run.incidents, {})
    assert shifted.false_alarms == len(run.incidents)
    assert shifted.precision == 0.0 and shifted.mean_delay_min is None


def test_cli_offline_demo(tmp_path, capsys):
    assert main(["demo", "--offline", "--hours", "6", "--out", str(tmp_path)]) == 0
    text = (tmp_path / "report.md").read_text()
    assert "Scorecard" in text and "Recommended actions" in text
    assert list((tmp_path / "charts").glob("INC-*.png"))
    assert "Recall 100%" in capsys.readouterr().out


def test_cli_rejects_tiny_fleet(capsys):
    assert main(["demo", "--offline", "--devices", "1"]) == 2


def test_cli_reports_refusal(tmp_path, monkeypatch, capsys):
    from edge_iot_agent import agent

    def refuse(self, incident, incidents):
        raise agent.InvestigationRefused("declined")

    monkeypatch.setattr(agent.Investigator, "__init__", lambda self, *a, **kw: None)
    monkeypatch.setattr(agent.Investigator, "diagnose", refuse)
    assert main(["demo", "--hours", "6", "--out", str(tmp_path)]) == 3
    assert "declined" in capsys.readouterr().out
