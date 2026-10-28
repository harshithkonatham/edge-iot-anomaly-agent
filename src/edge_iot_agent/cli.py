"""Command line: `iot-agent demo` simulates a motor fleet, detects incidents and diagnoses them."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import report, rules
from .detector import StreamingDetector
from .pipeline import detect, diagnose_all, score
from .simulator import FleetConfig, plan_faults, simulate
from .store import Store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="iot-agent", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Simulate, detect, diagnose and report")
    demo.add_argument("--devices", type=int, default=6)
    demo.add_argument("--hours", type=float, default=12)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--out", type=Path, default=Path("reports"))
    demo.add_argument("--db", default=":memory:", help="DuckDB file to keep the readings")
    demo.add_argument("--offline", action="store_true", help="Rule-based diagnosis, no API key")
    demo.add_argument("--model", default="claude-opus-5-5")
    demo.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh"])
    demo.add_argument("--max-incidents", type=int, default=10, help="Cap on Claude investigations")
    args = parser.parse_args(argv)

    if args.devices < 2 or args.hours < 3:
        print("Use at least 2 devices and 3 hours so faults fit after the warm-up period.")
        return 2

    cfg = FleetConfig(devices=args.devices, hours=args.hours, seed=args.seed)
    faults = plan_faults(cfg)
    run = detect(simulate(cfg, faults), Store(args.db), StreamingDetector())
    print(
        f"Streamed {run.readings} readings from {cfg.devices} motors: "
        f"{run.anomalies} anomalous readings → {len(run.incidents)} incidents"
    )

    if args.offline:
        engine = "offline rules"
        diagnoses = diagnose_all(run.incidents, lambda i: rules.diagnose(i, run.store))
    else:
        import anthropic

        from .agent import InvestigationRefused, Investigator

        engine = f"Claude investigator ({args.model}, effort {args.effort})"
        investigator = Investigator(run.store, run.baselines, model=args.model, effort=args.effort)
        by_id = {i.id: i for i in run.incidents}
        try:
            diagnoses = diagnose_all(
                run.incidents[: args.max_incidents], lambda i: investigator.diagnose(i, by_id)
            )
        except anthropic.AuthenticationError:
            print("No valid Anthropic credentials. Set ANTHROPIC_API_KEY or use --offline.")
            return 2
        except InvestigationRefused as exc:
            print(exc)
            return 3

    result = score(faults, run.incidents, diagnoses)
    charts = {i.id: report.plot_incident(run.store, i, args.out) for i in run.incidents}
    text = report.render(run.incidents, diagnoses, result, engine, charts, args.out)
    path = args.out / "report.md"
    path.write_text(text, encoding="utf-8")
    print(
        f"Recall {result.recall:.0%} · precision {result.precision:.0%} · "
        f"diagnosis accuracy {result.diagnosis_accuracy:.0%} → {path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
