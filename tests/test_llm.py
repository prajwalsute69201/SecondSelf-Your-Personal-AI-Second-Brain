"""Unit tests for lib/llm.py Groq integration module (Task 2.1.2).

Validates:
- Module import and default model configuration.
- Missing and placeholder credential rejection.
- Client initialization and singleton caching.
- Successful chat completions with mocked Groq API responses.
- Safe handling of empty or malformed API responses.
- API error handling (authentication, rate limits, connection errors).
- Strict credential leakage prevention in error reporting.
"""

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure secondself is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import groq
from groq import APIConnectionError, AuthenticationError, InternalServerError, RateLimitError
import lib.llm as llm


class TestLLMModule(unittest.TestCase):
    """Test suite for lib/llm.py."""

    def setUp(self) -> None:
        """Reset LLM client singleton before each test."""
        llm.reset_client()

    def tearDown(self) -> None:
        """Reset LLM client singleton after each test."""
        llm.reset_client()

    def test_01_module_import_and_default_model(self) -> None:
        """Requirement 1: Module imports successfully with correct model configuration."""
        self.assertTrue(hasattr(llm, "call_llm"))
        self.assertTrue(hasattr(llm, "get_client"))
        self.assertTrue(hasattr(llm, "DEFAULT_MODEL"))
        self.assertEqual(llm.DEFAULT_MODEL, "llama-3.1-8b-instant")
        self.assertEqual(llm.get_default_model(), "llama-3.1-8b-instant")

        # Test central model modification and restoration
        llm.set_default_model("llama-3.3-70b-versatile")
        self.assertEqual(llm.get_default_model(), "llama-3.3-70b-versatile")
        llm.set_default_model("llama-3.1-8b-instant")

    def test_02_missing_and_placeholder_credentials(self) -> None:
        """Requirement 2: Missing or placeholder GROQ_API_KEY raises actionable ValueError."""
        # Test completely missing key
        env_copy = os.environ.copy()
        env_copy.pop("GROQ_API_KEY", None)
        with patch.dict("os.environ", env_copy, clear=False), patch.object(llm, "_load_env"):
            with self.assertRaises(ValueError) as cm:
                llm.get_client(force_reload=True)
            self.assertIn("GROQ_API_KEY not found", str(cm.exception))

        # Test empty string key
        with patch.dict("os.environ", {"GROQ_API_KEY": "   "}), patch.object(llm, "_load_env"):
            with self.assertRaises(ValueError) as cm:
                llm.get_client(force_reload=True)
            self.assertIn("GROQ_API_KEY not found or empty", str(cm.exception))

        # Test template placeholder string
        with patch.dict("os.environ", {"GROQ_API_KEY": "your_key_here"}), patch.object(llm, "_load_env"):
            with self.assertRaises(ValueError) as cm:
                llm.get_client(force_reload=True)
            self.assertIn("default placeholder", str(cm.exception))

    def test_03_client_initialization_and_singleton(self) -> None:
        """Requirement 3: Client initializes properly and reuses singleton instance."""
        mock_key = "gsk_test_mock_dummy_key_1234567890"

        with patch.dict("os.environ", {"GROQ_API_KEY": mock_key}), patch.object(llm, "_load_env"):
            client1 = llm.get_client(force_reload=True)
            self.assertIsInstance(client1, groq.Groq)

            # Subsequent call must return the exact same instance (no duplicate creation)
            client2 = llm.get_client()
            self.assertIs(client1, client2)

            # Resetting clears the cached singleton
            llm.reset_client()
            client3 = llm.get_client()
            self.assertIsNot(client1, client3)

    def test_04_successful_mocked_response(self) -> None:
        """Requirement 4: call_llm executes chat completion and extracts assistant content."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "{\"para\": \"Projects\", \"tags\": [\"ml\"], \"summary\": \"Test note\"}"
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        prompt = "Classify this note content"
        system = "You are a PARA knowledge classifier."
        result = llm.call_llm(prompt, system=system, client=mock_client)

        self.assertEqual(result, mock_choice.message.content)
        mock_client.chat.completions.create.assert_called_once_with(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            model="llama-3.1-8b-instant",
            temperature=0.2,
        )

    def test_05_empty_and_malformed_responses(self) -> None:
        """Requirement 5: Empty choices or empty message content safely raises RuntimeError."""
        mock_client = MagicMock()

        # Case A: Empty choices list
        mock_response_no_choices = MagicMock()
        mock_response_no_choices.choices = []
        mock_client.chat.completions.create.return_value = mock_response_no_choices

        with self.assertRaises(RuntimeError) as cm:
            llm.call_llm("test prompt", client=mock_client)
        self.assertIn("empty choices list", str(cm.exception))

        # Case B: None message
        mock_response_none_msg = MagicMock()
        mock_choice_none_msg = MagicMock()
        mock_choice_none_msg.message = None
        mock_response_none_msg.choices = [mock_choice_none_msg]
        mock_client.chat.completions.create.return_value = mock_response_none_msg

        with self.assertRaises(RuntimeError) as cm:
            llm.call_llm("test prompt", client=mock_client)
        self.assertIn("missing message object", str(cm.exception))

        # Case C: Empty / whitespace content string
        mock_response_empty_str = MagicMock()
        mock_choice_empty_str = MagicMock()
        mock_choice_empty_str.message.content = "   "
        mock_response_empty_str.choices = [mock_choice_empty_str]
        mock_client.chat.completions.create.return_value = mock_response_empty_str

        with self.assertRaises(RuntimeError) as cm:
            llm.call_llm("test prompt", client=mock_client)
        self.assertIn("empty response content", str(cm.exception))

    def test_06_authentication_error_fails_immediately_without_secret_leakage(self) -> None:
        """Requirement 6: AuthenticationError terminates immediately without retrying or leaking key."""
        mock_client = MagicMock()
        secret_key_leak = "gsk_super_secret_user_key_99999"

        # Mock an AuthenticationError that embeds a sensitive key in its text
        auth_err = AuthenticationError(
            message=f"Invalid API key: {secret_key_leak}",
            response=MagicMock(status_code=401),
            body={"error": f"Invalid API key: {secret_key_leak}"},
        )
        mock_client.chat.completions.create.side_effect = auth_err

        with self.assertRaises(RuntimeError) as cm:
            llm.call_llm("test prompt", client=mock_client, max_retries=3)

        err_msg = str(cm.exception)
        # Authentication failure must not leak the raw key
        self.assertNotIn(secret_key_leak, err_msg)
        self.assertIn("authentication failed", err_msg.lower())
        # Authentication error must not retry (called only once)
        self.assertEqual(mock_client.chat.completions.create.call_count, 1)

    def test_07_rate_limit_retry_with_backoff(self) -> None:
        """Requirement 7: RateLimitError retries with exponential backoff and succeeds on retry."""
        mock_client = MagicMock()

        rate_limit_err = RateLimitError(
            message="Rate limit reached: 30 requests per minute",
            response=MagicMock(status_code=429),
            body={"error": "Rate limit reached"},
        )
        mock_success = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Recovered response"
        mock_success.choices = [mock_choice]

        # Fail twice with RateLimitError, then succeed on 3rd attempt
        mock_client.chat.completions.create.side_effect = [
            rate_limit_err,
            rate_limit_err,
            mock_success,
        ]

        result = llm.call_llm(
            "test prompt",
            client=mock_client,
            max_retries=3,
            initial_delay=0.001,
            backoff_factor=1.5,
        )

        self.assertEqual(result, "Recovered response")
        self.assertEqual(mock_client.chat.completions.create.call_count, 3)

    def test_08_connection_error_exhausted_retries_without_secret_leakage(self) -> None:
        """Requirement 8: APIConnectionError retries until max_retries without leaking tokens."""
        mock_client = MagicMock()
        secret_token = "gsk_leaked_connection_token_888"

        conn_err = APIConnectionError(
            message=f"Connection refused with Bearer {secret_token}",
            request=MagicMock(),
        )
        mock_client.chat.completions.create.side_effect = conn_err

        with self.assertRaises(RuntimeError) as cm:
            llm.call_llm(
                "test prompt",
                client=mock_client,
                max_retries=2,
                initial_delay=0.001,
            )

        err_msg = str(cm.exception)
        self.assertNotIn(secret_token, err_msg)
        self.assertIn("[REDACTED_API_KEY]", err_msg)
        self.assertIn("network/connection error after 2 attempts", err_msg)
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)

    def test_09_prompt_validation(self) -> None:
        """Requirement 9: Empty or whitespace-only prompt raises ValueError."""
        with self.assertRaises(ValueError):
            llm.call_llm("")

        with self.assertRaises(ValueError):
            llm.call_llm("   \n\t  ")

    def test_10_sanitize_error_message_helper(self) -> None:
        """Requirement 10: _sanitize_error_message properly scrubs credentials."""
        raw = "Failed on gsk_abc123XYZ and Bearer my_jwt_token_123 with api_key='gsk_xyz789'"
        cleaned = llm._sanitize_error_message(raw)

        self.assertNotIn("gsk_abc123XYZ", cleaned)
        self.assertNotIn("my_jwt_token_123", cleaned)
        self.assertNotIn("gsk_xyz789", cleaned)
        self.assertIn("[REDACTED_API_KEY]", cleaned)
        self.assertIn("Bearer [REDACTED]", cleaned)


if __name__ == "__main__":
    unittest.main()
