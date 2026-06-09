import unittest

from ai_scientist.perform_ideation_temp_free import parse_action_arguments


class IdeationParserTest(unittest.TestCase):
    def test_parse_action_arguments_ignores_repeated_action_after_json(self):
        arguments = """{"query": "confused deputy LLM agents"}
ACTION:
SearchSemanticScholar

ARGUMENTS:
{"query": "duplicate"}"""

        self.assertEqual(
            parse_action_arguments(arguments),
            {"query": "confused deputy LLM agents"},
        )

    def test_parse_action_arguments_extracts_json_code_block(self):
        arguments = """```json
{"idea": {"Name": "scopewash"}}
```"""

        self.assertEqual(
            parse_action_arguments(arguments),
            {"idea": {"Name": "scopewash"}},
        )


if __name__ == "__main__":
    unittest.main()
