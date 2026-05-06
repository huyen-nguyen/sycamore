"""Sycamore CLI.

Examples:
    # Run the ungrounded condition (7 evaluators).
    python -m sycamore.cli run --condition ungrounded

    # Run the grounded condition (the 7 EVALUATORS from step8_evaluators).
    python -m sycamore.cli run --condition grounded

    # Both conditions back to back.
    python -m sycamore.cli run --condition both

    # Smaller smoke run.
    python -m sycamore.cli run --condition ungrounded --n 2 --queries-per-modality 1

Provider:
    LLM_PROVIDER=anthropic python -m sycamore.cli run --condition grounded
    LLM_PROVIDER=openai    python -m sycamore.cli run --condition grounded
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llm_provider import get_fast_provider, get_provider

from .analysis import write_run_report
from .conditions import build_evaluators
from .geranium_client import GeraniumClient
from .protocol import ProtocolRunner

try:  # pretty logging if installed; harmless if not
    from rich.console import Console
    _console = Console()
    def _log(msg: str) -> None:
        _console.log(msg)
except Exception:
    def _log(msg: str) -> None:
        print(msg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sycamore")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run an evaluation condition end-to-end.")
    p_run.add_argument(
        "--condition", choices=["ungrounded", "grounded", "both"], required=True
    )
    p_run.add_argument("--n", type=int, default=7,
                       help="Number of ungrounded evaluators (default: 7).")
    p_run.add_argument("--queries-per-modality", type=int, default=3,
                       help="Queries per modality during exploration (default: 3).")
    p_run.add_argument("--k", type=int, default=5,
                       help="Top-k results per Geranium query (default: 5).")
    p_run.add_argument("--geranium-url", default="http://localhost:5001")
    p_run.add_argument("--out", default="data/sycamore_outputs")
    p_run.add_argument("--only", default=None,
                       help="Comma-separated abbreviations to include in grounded "
                            "condition (e.g., 'CB1,SE2'). Default: all.")
    p_run.add_argument("--collect-ranking", action="store_true", default=True)

    args = parser.parse_args(argv)
    if args.cmd == "run":
        return _cmd_run(args)
    parser.print_help()
    return 2


def _cmd_run(args) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    provider = get_provider()
    fast_provider = get_fast_provider()
    _log(f"[bold]LLM[/bold]: main={provider.name}; fast={fast_provider.name}")

    client = GeraniumClient(base_url=args.geranium_url)
    _log(f"[bold]Geranium[/bold]: {client.base_url}")

    # warm up the model so it's loaded before evaluators start
    _log("Warming up Geranium (loading model weights)...")
    try:
        client.search("Text", "warmup", k=1)
        _log("[green]Geranium ready.[/green]")
    except Exception as e:
        _log(f"[yellow]Warmup warning:[/yellow] {e} — continuing anyway")

    runner = ProtocolRunner(
        client=client,
        queries_per_modality=args.queries_per_modality,
        k=args.k,
        collect_modality_ranking=args.collect_ranking,
    )

    only_abbr = (
        [s.strip() for s in args.only.split(",") if s.strip()]
        if args.only else None
    )

    conditions = (
        ["ungrounded", "grounded"] if args.condition == "both" else [args.condition]
    )

    all_records = []
    for cond in conditions:
        evaluators = build_evaluators(
            cond,
            provider=provider,
            fast_provider=fast_provider,
            n_ungrounded=args.n,
            only_abbr=only_abbr,
        )
        _log(f"[bold]{cond}[/bold]: {len(evaluators)} evaluators.")
        for ev in evaluators:
            _log(f"  -> running {ev.identity.eid}")
            try:
                rec = runner.run(ev)
                all_records.append(rec)
            except Exception as e:
                _log(f"  [red]error[/red]: {ev.identity.eid}: {e}")

    paths = write_run_report(all_records, out_dir)
    _log(f"[green]Wrote summary[/green]: {paths['summary_md']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())