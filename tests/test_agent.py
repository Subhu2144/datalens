import pytest

from app.agent import (
    AgentError,
    _format_gemini_error,
)


def test_quota_error_is_converted_to_user_friendly_message():
    error = Exception(
        "429 RESOURCE_EXHAUSTED: You exceeded your current quota."
    )

    message = _format_gemini_error(error, "request")

    assert message == (
        "Gemini API quota is temporarily exhausted. "
        "Please try again later or check your Gemini API quota and billing settings."
    )


def test_quota_error_is_detected_case_insensitively():
    error = Exception("resource_exhausted")

    message = _format_gemini_error(error, "explanation request")

    assert "Gemini API quota is temporarily exhausted." in message
    assert "billing settings." in message


def test_non_quota_error_keeps_useful_error_context():
    error = Exception("Connection failed")

    message = _format_gemini_error(error, "request")

    assert message == "Gemini request failed: Connection failed"


def test_quota_error_can_be_raised_as_agent_error():
    error = Exception("429 RESOURCE_EXHAUSTED")

    with pytest.raises(AgentError, match="Gemini API quota is temporarily exhausted"):
        raise AgentError(_format_gemini_error(error, "request"))
