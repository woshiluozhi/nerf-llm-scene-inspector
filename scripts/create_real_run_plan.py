#!/usr/bin/env python
"""Create a real-scene run action plan from pipeline artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nerf_llm_scene_inspector.evaluation.real_run_plan import write_real_run_plan  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Pipeline run directory.")
    parser.add_argument(
        "--output",
        help="Output directory. Defaults to RUN_DIR/real_run_plan.",
    )
    parser.add_argument("--input", help="Raw video file or image directory for the intended real run.")
    parser.add_argument("--type", choices=["video", "images"], help="Input type for the intended real run.")
    parser.add_argument("--data", help="Processed data output directory for the intended real run.")
    parser.add_argument("--backend", choices=["lerf", "opennerf"], help="Target semantic backend.")
    parser.add_argument(
        "--variant",
        choices=["lerf", "lerf-lite", "lerf-big", "opennerf"],
        help="Target language-field training variant.",
    )
    parser.add_argument("--queries", help="Query YAML file to use for the real run.")
    parser.add_argument(
        "--submission-packet",
        help=(
            "Optional submission_packet.json to read, for example "
            "results/pipeline_runs/<scene>/submission_packet/submission_packet.json after finalization."
        ),
    )
    parser.add_argument("--output-root", default="results/pipeline_runs", help="Pipeline output root.")
    parser.add_argument(
        "--no-require-gpu",
        action="store_true",
        help="Do not mark CUDA/GPU-dependent steps as blocked in the generated plan.",
    )
    parser.add_argument("--repo-url", default="", help="Repository URL to include in sharing commands.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        plan = write_real_run_plan(
            args.run_dir,
            output_dir=args.output,
            input_path=args.input,
            input_type=args.type,
            processed_data=args.data,
            backend=args.backend,
            variant=args.variant,
            queries_path=args.queries,
            submission_packet_path=args.submission_packet,
            output_root=args.output_root,
            require_gpu=not args.no_require_gpu,
            repo_url=args.repo_url,
        )
    except Exception as exc:
        print(f"create_real_run_plan failed: {exc}", file=sys.stderr)
        return 1

    output = Path(args.output) if args.output else Path(args.run_dir) / "real_run_plan"
    print(json.dumps(plan.to_dict(), indent=2))
    print(f"\nWrote {output / 'real_run_plan.json'}")
    print(f"Wrote {output / 'real_run_plan.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
