import unittest
from unittest import mock

from ai_scientist.tools import semantic_scholar


class SemanticScholarRateLimitTest(unittest.TestCase):
    def setUp(self):
        semantic_scholar._last_s2_request_started_at = 0.0

    def test_s2_get_waits_between_consecutive_requests(self):
        response = mock.Mock()
        monotonic_values = [10.0, 10.0, 10.2, 11.05]

        with mock.patch.object(
            semantic_scholar.requests, "get", return_value=response
        ) as get:
            with mock.patch.object(
                semantic_scholar.time, "monotonic", side_effect=monotonic_values
            ):
                with mock.patch.object(semantic_scholar.time, "sleep") as sleep:
                    semantic_scholar._rate_limited_s2_get("https://example.test/one")
                    semantic_scholar._rate_limited_s2_get("https://example.test/two")

        self.assertEqual(get.call_count, 2)
        sleep.assert_called_once_with(1.0500000000000007)


if __name__ == "__main__":
    unittest.main()
