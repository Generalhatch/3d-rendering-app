"""CLI entry-point for the vectorize eval harness.

Run from the ``backend`` directory:

    python -m app.eval --scan demo/stevenson.laz \
                       --gt   demo/stevenson_gt.json \
                       --out  eval_runs/baseline

    python -m app.eval --compare eval_runs/baseline/metrics.json \
                       --against  eval_runs/2026-05-21/metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .harness import compare_metrics, run_eval
from .metrics import render_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.eval",
        description="Vectorize pipeline evaluation harness "
                    "(see backend/app/eval/__init__.py for what it scores).",
    )
    sub = parser.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="run the pipeline + score against ground truth")
    p_run.add_argument("--scan", required=True, type=Path)
    p_run.add_argument("--gt", required=True, type=Path)
    p_run.add_argument("--out", required=True, type=Path)
    p_run.add_argument(
        "--params", type=Path,
        help="optional JSON file with VectorizeParams overrides",
    )
    p_run.add_argument("--match-iou", type=float, default=0.30)

    p_cmp = sub.add_parser("compare", help="compare two metrics.json files")
    p_cmp.add_argument("--baseline", required=True, type=Path)
    p_cmp.add_argument("--against", required=True, type=Path)
    p_cmp.add_argument(
        "--regression-pct", type=float, default=2.0,
        help="metric drop > this percentage flags a regression",
    )

    args = parser.parse_args(argv)
    if args.cmd is None:
        # Convenience: allow `python -m app.eval --scan ... --gt ... --out ...`
        # without the explicit `run` subcommand.
        if "--scan" in (argv or sys.argv[1:]):
            return main(["run", *(argv or sys.argv[1:])])
        if "--compare" in (argv or sys.argv[1:]):
            return main(["compare", *(argv or sys.argv[1:])])
        parser.print_help()
        return 2

    if args.cmd == "run":
        overrides = None
        if args.params:
            overrides = json.loads(args.params.read_text())
        run_eval(
            scan_path=args.scan,
            gt_path=args.gt,
            out_dir=args.out,
            params_overrides=overrides,
            match_iou=args.match_iou,
        )
        return 0

    if args.cmd == "compare":
        passed, messages = compare_metrics(
            args.baseline, args.against,
            regression_pct=args.regression_pct,
        )
        for m in messages:
            print(m)
        print()
        print("RESULT:", "PASS" if passed else "FAIL")
        return 0 if passed else 1

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
