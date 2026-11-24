"""SecondSelf — Alias test runner for API tests (Task 4.2).

Ensures 'pytest tests/test_api.py' and 'pytest tests/test_ask_api.py' both execute
the complete serverless endpoint contract validation suite.
"""

from tests.test_ask_api import *  # noqa: F401, F403
