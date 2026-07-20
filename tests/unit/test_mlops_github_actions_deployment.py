import json
from unittest.mock import MagicMock
from unittest.mock import patch

from mlops.deployment import GitHubActionsDeploymentPipeline


def _proc(returncode=0, stdout="", stderr=""):
    return MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)


def _run_list(ids):
    return _proc(stdout=json.dumps([{"databaseId": i} for i in ids]))


def _run_view(status="completed", conclusion="success"):
    return _proc(stdout=json.dumps({"status": status, "conclusion": conclusion}))


def _pipeline(**overrides):
    defaults = dict(
        repo="acme/enterprise-rag-platform",
        test_workflow="ci.yml",
        poll_interval_seconds=0.0
    )
    defaults.update(overrides)
    return GitHubActionsDeploymentPipeline(**defaults)


def test_stage_with_no_workflow_configured_is_a_soft_skip():

    pipeline = GitHubActionsDeploymentPipeline(repo="acme/repo")

    result = pipeline.run_tests()

    assert result.success is True
    assert "skipped" in result.output


@patch("mlops.deployment.subprocess.run")
def test_run_tests_dispatches_polls_and_reports_success(mock_run):

    mock_run.side_effect = [
        _run_list([100]),               # ids before dispatch
        _proc(returncode=0),            # workflow run dispatch
        _run_list([100, 101]),          # ids after dispatch - 101 is new
        _run_view(conclusion="success")  # run view - completed
    ]
    pipeline = _pipeline()

    result = pipeline.run_tests()

    assert result.success is True
    assert "101" in result.output
    dispatch_call = mock_run.call_args_list[1].args[0]
    assert dispatch_call[:4] == ["gh", "workflow", "run", "ci.yml"]
    assert "--ref" in dispatch_call


@patch("mlops.deployment.subprocess.run")
def test_run_tests_reports_failure_when_dispatch_fails(mock_run):

    mock_run.side_effect = [
        _run_list([100]),
        _proc(returncode=1, stderr="workflow not found")
    ]
    pipeline = _pipeline()

    result = pipeline.run_tests()

    assert result.success is False
    assert "workflow not found" in result.output


@patch("mlops.deployment.subprocess.run")
def test_run_tests_reports_failure_on_non_success_conclusion(mock_run):

    mock_run.side_effect = [
        _run_list([100]),
        _proc(returncode=0),
        _run_list([100, 101]),
        _run_view(conclusion="failure")
    ]
    pipeline = _pipeline()

    result = pipeline.run_tests()

    assert result.success is False
    assert "failure" in result.output


@patch("mlops.deployment.subprocess.run")
def test_wait_for_new_run_times_out_when_no_new_run_appears(mock_run):

    mock_run.side_effect = [
        _run_list([100]),
        _proc(returncode=0)
    ]
    pipeline = _pipeline(poll_timeout_seconds=0.0)

    result = pipeline.run_tests()

    assert result.success is False
    assert "timed out" in result.output


@patch("mlops.deployment.subprocess.run")
def test_run_evaluation_uses_the_evaluation_workflow(mock_run):

    mock_run.side_effect = [
        _run_list([]),
        _proc(returncode=0),
        _run_list([200]),
        _run_view()
    ]
    pipeline = _pipeline(evaluation_workflow="evaluate.yml")

    result = pipeline.run_evaluation()

    assert result.success is True
    dispatch_call = mock_run.call_args_list[1].args[0]
    assert "evaluate.yml" in dispatch_call


@patch("mlops.deployment.subprocess.run")
def test_run_experiment_passes_experiment_id_as_workflow_input(mock_run):

    mock_run.side_effect = [
        _run_list([]),
        _proc(returncode=0),
        _run_list([300]),
        _run_view()
    ]
    pipeline = _pipeline(experiment_workflow="experiment.yml")

    pipeline.run_experiment("exp-42")

    dispatch_call = mock_run.call_args_list[1].args[0]
    assert "-f" in dispatch_call
    assert "experiment_id=exp-42" in dispatch_call


@patch("mlops.deployment.subprocess.run")
def test_deploy_passes_asset_id_and_target_stage_as_workflow_inputs(mock_run):

    mock_run.side_effect = [
        _run_list([]),
        _proc(returncode=0),
        _run_list([400]),
        _run_view()
    ]
    pipeline = _pipeline(deploy_workflow="deploy.yml")

    pipeline.deploy("embedding_model:hashing:1.0", "production")

    dispatch_call = mock_run.call_args_list[1].args[0]
    assert "asset_id=embedding_model:hashing:1.0" in dispatch_call
    assert "target_stage=production" in dispatch_call


@patch("mlops.deployment.subprocess.run")
def test_recent_run_ids_returns_empty_set_on_gh_failure(mock_run):

    mock_run.return_value = _proc(returncode=1, stderr="gh: not authenticated")
    pipeline = _pipeline()

    assert pipeline._recent_run_ids("ci.yml") == set()
