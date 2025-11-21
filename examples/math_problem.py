import asyncio
import os
import sys
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc_agi.interfaces import (
    ExecutionResult,
    Feedback,
    FeedbackGenerator,
    Problem,
    PromptGenerator,
    Sandbox,
)
from arc_agi.solve_parallel_coding import solve_parallel_coding
from arc_agi.types import ExpertConfig


class MathProblem(Problem):
    def __init__(self, question: str, answer: int):
        self.question = question
        self.answer = answer

    @property
    def description(self) -> str:
        return self.question

    def get_train_examples(self) -> list:
        return [(self.question, self.answer)]

    def get_test_examples(self) -> list:
        return []


class LocalPythonSandbox(Sandbox):
    async def run(
        self, code: str, input_data: Any, timeout_s: float = 1.5
    ) -> ExecutionResult:
        # Very unsafe sandbox for demonstration only!
        # Captures stdout
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        try:
            # We expect the code to print the answer
            with redirect_stdout(f):
                exec(code, {}, {})
            output = f.getvalue().strip()
            return ExecutionResult(success=True, output=output, error=None)
        except Exception as e:
            return ExecutionResult(success=False, output="", error=str(e))


class MathFeedbackGenerator(FeedbackGenerator):
    def generate(self, result: ExecutionResult, expected_output: Any) -> Feedback:
        # Used by generate_batch internally if needed
        if not result.success:
            return Feedback(score=0.0, text=f"Error: {result.error}", is_correct=False)

        try:
            val = int(result.output)
            if val == expected_output:
                return Feedback(score=1.0, text="Correct!", is_correct=True)
            else:
                return Feedback(
                    score=0.0,
                    text=f"Incorrect. Expected {expected_output}, got {val}",
                    is_correct=False,
                )
        except ValueError:
            return Feedback(
                score=0.0, text="Output was not an integer", is_correct=False
            )

    def generate_batch(
        self, results: list[ExecutionResult], train_in, train_out
    ) -> tuple[str, float]:
        total_score = 0.0
        feedback_texts = []
        for res, expected in zip(results, train_out):
            fb = self.generate(res, expected)
            total_score += fb.score
            feedback_texts.append(fb.text)

        avg_score = total_score / len(results) if results else 0.0
        return "\n".join(feedback_texts), avg_score


class MathPromptGenerator(PromptGenerator):
    def generate_solver_prompt(self, problem: Problem) -> str:
        return f"""
You are a python coding assistant.
Solve the following math problem by writing a python script that prints the answer.
Problem: {problem.description}
Do not output anything else but the python code.
"""

    def generate_feedback_prompt(self, previous_solutions: list[dict]) -> str:
        return ""


async def mock_llm(*args, **kwargs):
    return "```python\nprint(30)\n```", 0.1, None, 5


async def main():
    # Patch the LLM to avoid needing API keys for this example
#    import arc_agi.solve_coding

#    arc_agi.solve_coding.llm = mock_llm

    problem = MathProblem(question="What is 10 + 20?", answer=30)
    # Choose sandbox based on env var, default to MCPSandbox
    from arc_agi.sandbox import MCPSandbox

    print("Using MCPSandbox (secure default)...")
    sandbox = MCPSandbox()
    feedback_generator = MathFeedbackGenerator()
    prompt_generator = MathPromptGenerator()

    config: ExpertConfig = {
        "llm_id": "openrouter/x-ai/grok-4-1-fast",
        "max_iterations": 1,
        "solver_temperature": 0.0,
        "selection_probability": 1.0,
        "seed": 42,
        "return_best_result": True,
        "use_new_voting": False,
        "count_failed_matches": False,
        "iters_tiebreak": False,
        "low_to_high_iters": False,
        "timeout_s": 2.0,
        # Missing keys filled with defaults
        "solver_prompt": "",
        "feedback_prompt": "",
        "max_solutions": 1,
        "shuffle_examples": False,
        "improving_order": False,
        "request_timeout": 60,
        "max_total_timeouts": 5,
        "max_total_time": None,
        "num_experts": 1,
        "per_iteration_retries": 0,
    }

    print(f"Solving: {problem.description}")
    results = await solve_parallel_coding(
        problem=problem,
        sandbox=sandbox,
        feedback_generator=feedback_generator,
        prompt_generator=prompt_generator,
        expert_configs=[config],
    )

    print("Results:")
    for res in results:
        print(res)


if __name__ == "__main__":
    asyncio.run(main())
