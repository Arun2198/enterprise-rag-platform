"""
CLI entry point for the retrieval evaluation framework.

    uv run python evaluation/run_eval.py \
        --dataset evaluation/golden_dataset.json \
        --k 5 \
        --json

    # compare against a previous run
    uv run python evaluation/run_eval.py \
        --dataset evaluation/golden_dataset.json \
        --baseline evaluation/reports/ai-rmf-1stdraft_20260101_000000.json
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from app.config import load_settings  # noqa: E402
from evaluation import report  # noqa: E402
from evaluation.benchmark import BenchmarkConfig  # noqa: E402
from evaluation.benchmark import BenchmarkRunner  # noqa: E402
from evaluation.dataset import DatasetValidationError  # noqa: E402
from evaluation.dataset import load_dataset  # noqa: E402
from evaluation.runner import DEFAULT_K_VALUES  # noqa: E402


def parse_args(
    argv: list[str] | None = None
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run retrieval evaluation against a golden dataset."
    )
    parser.add_argument(
        "--dataset",
        default="evaluation/golden_dataset.json",
        help="Path to a golden dataset JSON file"
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=None,
        help="K values to evaluate (default: 1 3 5 10, plus EVALUATION_DEFAULT_K)"
    )
    parser.add_argument(
        "--provider",
        default="hashing",
        help="Embedding provider: 'hashing' or any sentence-transformers model name"
    )
    parser.add_argument(
        "--reranker",
        action="store_true",
        help="Enable cross-encoder reranking"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Report output directory (default: EVALUATION_REPORT_DIR)"
    )
    parser.add_argument("--json", action="store_true", help="Write a JSON report")
    parser.add_argument("--csv", action="store_true", help="Write a CSV report")
    parser.add_argument(
        "--baseline",
        default=None,
        help="Path to a baseline JSON report to compare against"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=report.DEFAULT_REGRESSION_THRESHOLD,
        help="Regression threshold as a fraction (default 0.02 = 2%%)"
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None
) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    settings = load_settings()

    if not settings.evaluation_enabled:
        print("EVALUATION_ENABLED=false - nothing to do.")
        return 0

    try:
        dataset = load_dataset(args.dataset)
    except DatasetValidationError as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 1

    k_values = sorted(set(args.k or list(DEFAULT_K_VALUES) + [settings.evaluation_default_k]))
    config = BenchmarkConfig(
        label=args.provider,
        embedder_name=args.provider,
        use_reranker=args.reranker,
        k_values=k_values
    )

    ((_, evaluation_report),) = BenchmarkRunner(dataset).run([config])

    print(report.render_console(evaluation_report))

    output_dir = args.output or settings.evaluation_report_dir
    exit_code = 0

    if args.json:
        report.write_json_report(evaluation_report, output_dir)

    if args.csv:
        report.write_csv_report(evaluation_report, output_dir)

    if args.baseline:
        baseline_report = report.load_json_report(args.baseline)
        comparison = report.compare_reports(
            current=evaluation_report,
            baseline=baseline_report,
            threshold=args.threshold
        )
        print()
        print(report.render_comparison_console(comparison))

        if comparison.has_regressions:
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
