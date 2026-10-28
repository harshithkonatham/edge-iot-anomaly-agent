"""Markdown incident report with a chart per incident and a ground-truth scorecard."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from .detector import Incident  # noqa: E402
from .models import Diagnosis  # noqa: E402
from .pipeline import Score  # noqa: E402
from .store import Store  # noqa: E402

SEVERITY_ICON = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}
UNITS = {"temperature_c": "°C", "vibration_mm_s": "mm/s", "current_a": "A"}


def plot_incident(store: Store, incident: Incident, out_dir: Path) -> Path:
    pad = timedelta(minutes=60)
    data = store.series(incident.device_id, incident.start - pad, incident.end + pad)
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    for ax, (sensor, unit) in zip(axes, UNITS.items(), strict=True):
        part = data[data["sensor"] == sensor]
        ax.plot(part["ts"], part["value"], lw=0.8, color="#4C78A8")
        ax.axvspan(incident.start, incident.end, color="#E45756", alpha=0.15)
        ax.set_ylabel(f"{sensor.split('_')[0]} ({unit})", fontsize=8)
    axes[0].set_title(f"{incident.id} · {incident.device_id} (shaded: incident window)")
    fig.autofmt_xdate()
    fig.tight_layout()
    path = out_dir / "charts" / f"{incident.id}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def render(
    incidents: list[Incident],
    diagnoses: dict[str, Diagnosis],
    result: Score,
    engine: str,
    charts: dict[str, Path],
    out_dir: Path,
) -> str:
    delay = result.mean_delay_min
    lines = [
        "# Motor fleet incident report",
        "",
        f"**Engine:** {engine}",
        "",
        "## Scorecard (against injected faults)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Faults injected | {result.faults} |",
        f"| Incidents raised | {result.incidents} |",
        f"| Detection recall | {result.recall:.0%} |",
        f"| Detection precision | {result.precision:.0%} ({result.false_alarms} false alarms) |",
        f"| Mean detection delay | {'n/a' if delay is None else f'{delay:.0f} min'} |",
        f"| Diagnosis accuracy | {result.diagnosis_accuracy:.0%} "
        f"({result.correct_causes}/{result.diagnosed}) |",
        "",
        "| Motor | Injected fault | Incident | Delay (min) | Diagnosed as |",
        "|---|---|---|---|---|",
    ]
    for row in result.rows:
        ok = "✅" if row["diagnosed"] == row["expected"] else "❌"
        diagnosed = f"{ok} {row['diagnosed']}" if row["diagnosed"] else "-"
        lines.append(
            f"| {row['device']} | {row['fault']} | {row['incident'] or 'missed'} "
            f"| {row['delay_min'] if row['delay_min'] is not None else '-'} | {diagnosed} |"
        )
    lines += ["", "## Incidents", ""]
    for incident in incidents:
        d = diagnoses.get(incident.id)
        window = f"{incident.start:%H:%M}-{incident.end:%H:%M}"
        lines += [f"### {incident.id} · {incident.device_id} · {window}", ""]
        if d:
            lines += [
                f"{SEVERITY_ICON[d.severity]} **{d.likely_cause.replace('_', ' ')}** "
                f"(severity {d.severity}, confidence {d.confidence})",
                "",
                d.summary,
                "",
                "**Evidence**",
                *[f"- {e}" for e in d.evidence],
                "",
                "**Recommended actions**",
                *[f"{n}. {a}" for n, a in enumerate(d.recommended_actions, 1)],
                "",
            ]
        if incident.id in charts:
            lines += [f"![{incident.id}]({charts[incident.id].relative_to(out_dir)})", ""]
    return "\n".join(lines)
