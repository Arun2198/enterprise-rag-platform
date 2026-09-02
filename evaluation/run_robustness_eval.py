"""
CLI entry point for the robustness evaluation (unanswerable/adversarial
queries) - see evaluation.robustness for why this is a separate dataset
and runner from the golden-dataset retrieval eval (run_eval.py).

    uv run python evaluation/run_robustness_eval.py \
        --dataset evaluation/robustness_dataset.json
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from app.config import load_settings  # noqa: E402
from app.service_factory import build_rag_service  # noqa: E402
from app.services.rag_service import ABSTENTION_MESSAGE  # noqa: E402
from evaluation.robustness import RobustnessDatasetError  # noqa: E402
from evaluation.robustness import load_robustness_dataset  # noqa: E402
from evaluation.robustness import run_robustness_eval  # noqa: E402


def parse_args(
    argv: list[str] | None = None
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the robustness (unanswerable/adversarial) evaluation."
    )
    parser.add_argument("--dataset", required=True, help="Path to a robustness dataset JSON file.")
    parser.add_argument(
        "--source-document",
        action="append",
        default=[],
        help="File to ingest before running (repeatable). Defaults to "
             "sample_documents/AI-RMF-1stdraft.pdf if omitted."
    )
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None
) -> int:
    args = parse_args(argv)

    try:
        dataset = load_robustness_dataset(args.dataset)
    except RobustnessDatasetError as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 1

    settings = load_settings()
    service = build_rag_service(settings)
    source_documents = args.source_document or ["sample_documents/AI-RMF-1stdraft.pdf"]
    document_ids = [Path(path).stem for path in source_documents]
    ingest_result = service.ingest(source_documents, document_ids=document_ids)

    if ingest_result.errors:
        for error in ingest_result.errors:
            print(f"ingest error: {error}", file=sys.stderr)
        return 1

    report = run_robustness_eval(service, dataset, ABSTENTION_MESSAGE)

    print(f"{report.dataset_name}: {report.passed_count}/{report.total} passed "
          f"({report.pass_rate:.0%})")

    for result in report.results:
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {result.case_id} ({result.category}) "
              f"observed={result.observed_action} - {result.detail}")

    return 0 if report.pass_rate == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
