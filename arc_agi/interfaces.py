from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ExecutionResult:
    success: bool
    output: str
    error: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class Feedback:
    score: float
    text: str
    is_correct: bool


class Problem(ABC):
    @property
    @abstractmethod
    def description(self) -> str:
        """Textual description of the problem to be solved."""
        pass

    @abstractmethod
    def get_train_examples(self) -> list[Any]:
        """Return training examples/test cases."""
        pass

    @abstractmethod
    def get_test_examples(self) -> list[Any]:
        """Return held-out test examples."""
        pass


class Sandbox(ABC):
    @abstractmethod
    async def run(
        self, code: str, input_data: Any, timeout_s: float = 1.5
    ) -> ExecutionResult:
        """Execute code with the given input data."""
        pass


class FeedbackGenerator(ABC):
    @abstractmethod
    def generate(self, result: ExecutionResult, expected_output: Any) -> Feedback:
        """Generate feedback by comparing execution result with expected output."""
        pass

    @abstractmethod
    def generate_batch(
        self, results: list[ExecutionResult], train_in: Any, train_out: Any
    ) -> tuple[str, float]:
        pass


class PromptGenerator(ABC):
    @abstractmethod
    def generate_solver_prompt(self, problem: Problem) -> str:
        """Generate the initial system prompt for the solver."""
        pass

    @abstractmethod
    def generate_feedback_prompt(self, previous_solutions: list[dict]) -> str:
        """Generate the prompt containing feedback on previous attempts."""
        pass
