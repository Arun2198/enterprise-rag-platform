from unittest.mock import MagicMock
from unittest.mock import patch

from app.aws_client_factory import build_boto3_client


@patch("app.aws_client_factory.boto3")
def test_build_boto3_client_passes_region_and_service(mock_boto3):

    build_boto3_client("bedrock-runtime", region_name="us-east-1")

    args, kwargs = mock_boto3.client.call_args
    assert args[0] == "bedrock-runtime"
    assert kwargs["region_name"] == "us-east-1"


@patch("app.aws_client_factory.boto3")
def test_build_boto3_client_uses_standard_retry_mode_with_the_given_max_attempts(mock_boto3):

    build_boto3_client("s3", region_name="us-east-1", max_attempts=5)

    config = mock_boto3.client.call_args.kwargs["config"]
    assert config.retries["mode"] == "standard"
    assert config.retries["max_attempts"] == 5


@patch("app.aws_client_factory.boto3")
def test_build_boto3_client_sets_connect_and_read_timeouts(mock_boto3):

    build_boto3_client("sqs", region_name="us-east-1", connect_timeout=2.0, read_timeout=15.0)

    config = mock_boto3.client.call_args.kwargs["config"]
    assert config.connect_timeout == 2.0
    assert config.read_timeout == 15.0


def test_build_boto3_client_returns_a_real_client_object():
    client = build_boto3_client("sts", region_name="us-east-1")

    assert client.meta.service_model.service_name == "sts"
