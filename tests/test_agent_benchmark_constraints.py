import unittest
from types import SimpleNamespace

from ai_scientist.treesearch.parallel_agent import MinimalAgent


def make_agent(require_llm_agent_calls=True):
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(
            agent_benchmark_only=True,
            require_llm_agent_calls=require_llm_agent_calls,
        )
    )
    return MinimalAgent(task_desc="{}", cfg=cfg)


class AgentBenchmarkConstraintTest(unittest.TestCase):
    def test_requires_experiment_data_artifact(self):
        agent = make_agent(require_llm_agent_calls=False)

        error = agent.validate_generated_code("print('metrics')")

        self.assertIsNotNone(error)
        self.assertIn("experiment_data.npy", error)

    def test_requires_openai_agent_calls_when_enabled(self):
        agent = make_agent(require_llm_agent_calls=True)
        deterministic_code = """
import os
working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)
np.save(os.path.join(working_dir, 'experiment_data.npy'), {})
"""

        error = agent.validate_generated_code(deterministic_code)

        self.assertIsNotNone(error)
        self.assertIn("OpenAI SDK", error)

    def test_accepts_bounded_openai_agent_benchmark_code(self):
        agent = make_agent(require_llm_agent_calls=True)
        openai_code = """
import os
from openai import OpenAI
working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)
if not os.getenv('OPENAI_API_KEY'):
    raise RuntimeError('OPENAI_API_KEY is required')
client = OpenAI()
response = client.chat.completions.create(
    model=os.getenv('AI_SCIENTIST_AGENT_BENCHMARK_MODEL', 'gpt-5.5'),
    messages=[{'role': 'user', 'content': 'Return JSON'}],
    temperature=1.0,
    max_completion_tokens=512,
)
np.save(os.path.join(working_dir, 'experiment_data.npy'), {})
"""

        self.assertIsNone(agent.validate_generated_code(openai_code))


if __name__ == "__main__":
    unittest.main()
