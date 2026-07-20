import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_EVAL_SCRIPT = REPO_ROOT / "evaluation" / "run_eval.py"


def _build_dataset_file(tmp_path):
    doc_path = tmp_path / "policy.md"
    doc_path.write_text(
        "# Leave Policy\n"
        "Employees receive twenty days of paid leave annually for full time staff. "
        "Contractors receive ten days of leave under this same policy. "
        "All leave requests must be submitted two weeks in advance of travel.",
        encoding="utf-8"
    )

    dataset_path = tmp_path / "dataset.json"
    posix_doc_path = str(doc_path).replace("\\", "/")
    dataset_path.write_text(
        json.dumps({
            "name": "cli-fixture",
            "source_documents": [posix_doc_path],
            "queries": [
                {
                    "id": "Q1",
                    "query": "How many leave days do contractors receive?",
                    "relevant_chunk_ids": ["policy:0"]
                }
            ]
        }),
        encoding="utf-8"
    )
    return str(dataset_path)


def _run_cli(args, cwd=None):
    return subprocess.run(
        [sys.executable, str(RUN_EVAL_SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=cwd or REPO_ROOT,
        timeout=60
    )


def test_cli_runs_and_prints_console_report(tmp_path):

    dataset_path = _build_dataset_file(tmp_path)

    result = _run_cli(["--dataset", dataset_path, "--k", "1", "3"])

    assert result.returncode == 0
    assert "cli-fixture" in result.stdout
    assert "Recall@1" in result.stdout


def test_cli_writes_json_and_csv_reports(tmp_path):

    dataset_path = _build_dataset_file(tmp_path)
    output_dir = tmp_path / "reports"

    result = _run_cli([
        "--dataset", dataset_path,
        "--k", "1",
        "--json", "--csv",
        "--output", str(output_dir)
    ])

    assert result.returncode == 0
    written = list(output_dir.glob("*.json")) + list(output_dir.glob("*.csv"))
    assert len(written) == 2


def test_cli_missing_dataset_returns_error_exit_code():

    result = _run_cli(["--dataset", "does/not/exist.json"])

    assert result.returncode == 1
    assert "error" in result.stderr.lower()


def test_cli_generation_flag_prints_layer_2_section(tmp_path):

    dataset_path = _build_dataset_file(tmp_path)

    result = _run_cli(["--dataset", dataset_path, "--k", "1", "--generation", "extractive"])

    assert result.returncode == 0
    assert "Generation quality (Layer 2)" in result.stdout
    assert "groundedness" in result.stdout


def test_cli_system_metrics_flag_prints_layer_3_section(tmp_path):

    dataset_path = _build_dataset_file(tmp_path)

    result = _run_cli(["--dataset", dataset_path, "--k", "1", "--system-metrics"])

    assert result.returncode == 0
    assert "System metrics (Layer 3)" in result.stdout
    assert "run_duration_seconds" in result.stdout


def test_cli_track_and_trend_record_and_print_history(tmp_path):

    dataset_path = _build_dataset_file(tmp_path)
    history_path = tmp_path / "history.json"

    first = _run_cli([
        "--dataset", dataset_path, "--k", "1",
        "--track", "--track-path", str(history_path)
    ])
    assert first.returncode == 0
    assert history_path.exists()

    second = _run_cli([
        "--dataset", dataset_path, "--k", "1",
        "--track", "--track-path", str(history_path),
        "--trend", "5"
    ])

    assert second.returncode == 0
    assert "recall@1" in second.stdout
    history = json.loads(history_path.read_text(encoding="utf-8"))
    assert len(history) == 2


def test_cli_baseline_comparison_reports_no_regression_for_identical_runs(tmp_path):

    dataset_path = _build_dataset_file(tmp_path)
    output_dir = tmp_path / "reports"

    first = _run_cli([
        "--dataset", dataset_path,
        "--k", "1",
        "--json",
        "--output", str(output_dir)
    ])
    assert first.returncode == 0
    baseline_path = next(output_dir.glob("*.json"))

    second = _run_cli([
        "--dataset", dataset_path,
        "--k", "1",
        "--baseline", str(baseline_path)
    ])

    assert second.returncode == 0
    assert "No regressions" in second.stdout
