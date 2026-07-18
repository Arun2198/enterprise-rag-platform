import sys
from unittest.mock import MagicMock
from unittest.mock import patch

from mlops.deployment import LocalDeploymentPipeline


def test_deploy_records_intent_without_side_effects():

    pipeline = LocalDeploymentPipeline()

    result = pipeline.deploy("embedding_model:hashing:1.0", "staging")

    assert result.stage == "deploy"
    assert result.success is True
    assert "hashing:1.0" in result.output
    assert "staging" in result.output


def test_run_stage_captures_real_success_and_output(tmp_path):

    pipeline = LocalDeploymentPipeline(repo_root=tmp_path)

    result = pipeline._run_stage("test", [sys.executable, "-c", "print('hello from stage')"])

    assert result.success is True
    assert "hello from stage" in result.output
    assert result.duration_seconds >= 0.0


def test_run_stage_captures_real_failure():

    pipeline = LocalDeploymentPipeline()

    result = pipeline._run_stage("test", [sys.executable, "-c", "import sys; sys.exit(1)"])

    assert result.success is False


@patch("mlops.deployment.subprocess.run")
def test_run_tests_invokes_pytest(mock_run):

    mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
    pipeline = LocalDeploymentPipeline()

    result = pipeline.run_tests()

    assert result.success is True
    command = mock_run.call_args.args[0]
    assert "pytest" in command


@patch("mlops.deployment.subprocess.run")
def test_run_evaluation_invokes_eval_cli(mock_run):

    mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
    pipeline = LocalDeploymentPipeline()

    result = pipeline.run_evaluation()

    assert result.success is True
    command = mock_run.call_args.args[0]
    assert "evaluation/run_eval.py" in command


@patch("mlops.deployment.subprocess.run")
def test_run_experiment_uses_named_output_directory(mock_run):

    mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
    pipeline = LocalDeploymentPipeline()

    pipeline.run_experiment("exp-001")

    command = mock_run.call_args.args[0]
    assert "evaluation/reports/exp-001" in command


@patch("mlops.deployment.subprocess.run")
def test_failed_stage_reports_failure(mock_run):

    mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="boom")
    pipeline = LocalDeploymentPipeline()

    result = pipeline.run_tests()

    assert result.success is False
    assert "boom" in result.output
