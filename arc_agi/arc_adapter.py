import json
import string
from typing import Any, Optional

import numpy as np

from arc_agi.interfaces import (
    ExecutionResult,
    Feedback,
    FeedbackGenerator,
    Problem,
    PromptGenerator,
    Sandbox,
)
from arc_agi.sandbox import run as original_run


class ARCProblem(Problem):
    def __init__(self, train_in, train_out, test_in, problem_id=None):
        self.train_in = train_in
        self.train_out = train_out
        self.test_in = test_in
        self.problem_id = problem_id

    @property
    def description(self) -> str:
        return ""

    def get_train_examples(self) -> list[Any]:
        return list(zip(self.train_in, self.train_out))

    def get_test_examples(self) -> list[Any]:
        return self.test_in


class ARCSandbox(Sandbox):
    async def run(
        self, code: str, input_data: Any, timeout_s: float = 1.5
    ) -> ExecutionResult:
        ok, out_str = await original_run(code, input_data, timeout_s=timeout_s)
        return ExecutionResult(
            success=ok,
            output=out_str if ok else "",
            error=None if ok else (out_str or "Execution failed."),
        )


class ARCFeedbackGenerator(FeedbackGenerator):
    def generate(self, result: ExecutionResult, expected_output: Any) -> Feedback:
        # Not used in current batch flow
        return Feedback(score=0.0, text="Not implemented", correctness=False)

    def generate_batch(
        self, results: list[ExecutionResult], train_in, train_out
    ) -> tuple[str, float]:
        run_results = []
        for i, r in enumerate(results):
            soft_score = 0.0
            if r.success:
                soft_score = 1.0
            elif r.output:
                arr = _json_to_ndarray(r.output)
                if arr is not None:
                    truth = np.array(train_out[i])
                    soft_score = _soft_score(arr, truth)

            run_results.append(
                {
                    "success": r.success,
                    "output": r.output,
                    "error": r.error,
                    "soft_score": soft_score,
                    "code": "",
                }
            )

        return _build_feedback(run_results, train_in, train_out)


class ARCPromptGenerator(PromptGenerator):
    def __init__(self, solver_prompt_template: str, feedback_prompt_template: str):
        self.solver_prompt_template = solver_prompt_template
        self.feedback_prompt_template = feedback_prompt_template

    def generate_solver_prompt(self, problem: Problem) -> str:
        if not isinstance(problem, ARCProblem):
            raise ValueError("ARCPromptGenerator only works with ARCProblem")

        example = _make_example(problem.train_in, problem.train_out, problem.test_in)
        problem_str = format_problem(example, shuffle=False)

        return self._build_prompt(self.solver_prompt_template, problem=problem_str)

    def generate_feedback_prompt(self, previous_solutions: list[dict]) -> str:
        examples_block = create_examples(
            previous_solutions, max_examples=5, improving_order=True
        )
        return "\n\n" + self._build_prompt(
            self.feedback_prompt_template, feedback=examples_block
        )

    def _build_prompt(self, base_prompt: str, **fields: str) -> str:
        s = base_prompt
        for k, v in fields.items():
            s = s.replace(f"$${k}$$", v)
        return s


# --- Helper Functions ---


def create_examples(solutions, max_examples=3, improving_order: bool = False):
    template = string.Template("""
<solution_$index>
<solution_code>
```python
$code
```
</solution_code>
<solution_evaluation>
$feedback
</solution_evaluation>
<solution_score>
$score
</solution_score>
</solution_$index>
""")
    if not solutions:
        return ""
    scores = [x["score"] for x in solutions]
    inds = np.argsort(scores)[::-1]
    if improving_order:
        inds = inds[::-1]
    inds = inds[: min(max_examples, len(inds))]

    blocks: list[str] = []
    for k, idx in enumerate(inds, start=1):
        e = solutions[idx]
        blocks.append(
            template.substitute(
                index=k,
                code=e["code"],
                feedback=e["feedback"],
                score=f"{e['score']:.2f}",
            )
        )
    return "\n".join(blocks)


def _array_diff(arr1: np.ndarray, arr2: np.ndarray) -> str:
    rows, cols = arr1.shape
    out = []
    for i in range(rows):
        row = []
        for j in range(cols):
            if arr1[i, j] == arr2[i, j]:
                row.append(str(int(arr1[i, j])))
            else:
                row.append(f"{int(arr1[i, j])}/{int(arr2[i, j])}")
        out.append(" ".join(row))
    return "\n".join(out)


def _soft_score(pred: np.ndarray, truth: np.ndarray) -> float:
    if pred.shape != truth.shape:
        return 0.0
    if truth.size == 0:
        return 1.0
    raw = np.mean(pred == truth)
    return float(np.nan_to_num(raw, posinf=0.0, neginf=0.0))


def _json_to_ndarray(s: str) -> Optional[np.ndarray]:
    try:
        obj = json.loads(s)
        arr = np.array(obj)
        if arr.ndim < 2:
            arr = np.expand_dims(arr, axis=list(range(2 - arr.ndim)))
        return arr.astype(int, copy=False)
    except Exception:
        return None


def _make_example(train_in, train_out, test_in) -> dict[str, Any]:
    train = [
        {"input": iin, "output": oout}
        for iin, oout in zip(train_in, train_out, strict=True)
    ]
    test = [{"input": iin} for iin in test_in]
    return {"train": train, "test": test}


def format_problem(
    problem: dict[str, Any],
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> str:
    train = list(problem["train"])
    test = list(problem["test"])

    if shuffle and len(train) > 1:
        rng = np.random.default_rng(seed if seed is not None else 0)
        perm = rng.permutation(len(train))
        train = [train[i] for i in perm]

    example_str = ""
    challenge_str = ""

    for example_num, example in enumerate(train, start=1):
        example_str += f"""
Example #{example_num}
Input:
<Diagram>
{_example_to_diagram(example["input"])}
</Diagram>

Output:
<Diagram>
{_example_to_diagram(example["output"])}
</Diagram>
"""

    for challenge_num, challenge in enumerate(test, start=1):
        challenge_str += f"""
Challenge #{challenge_num}
Input:
<Diagram>
{_example_to_diagram(challenge["input"])}
</Diagram>
"""

    return example_str + challenge_str


def _example_to_diagram(example: list[list[int]] | np.ndarray) -> str:
    diagram = ""
    for row in example:
        row_str = " ".join([str(col) for col in row]) + "\n"
        diagram += row_str
    return diagram[:-1]


def _parse_json_array_no_expand(s: str) -> Optional[np.ndarray]:
    try:
        return np.array(json.loads(s))
    except Exception:
        return None


def _build_feedback(
    train_results: list[dict], train_in, train_out
) -> tuple[str, float]:
    feedback_parts: list[str] = []
    per_example_scores: list[float] = []

    for i, rr in enumerate(train_results):
        if rr["success"]:
            feedback_parts.append(f"Solves Example #{i + 1} correctly. ")
            per_example_scores.append(1.0)
            continue

        msg_lines: list[str] = [f"Solves Example #{i + 1} incorrectly. "]

        pred_raw = _parse_json_array_no_expand(rr["output"]) if rr["output"] else None
        truth = np.array(train_out[i])

        if pred_raw is None:
            per_example_scores.append(0.0)
            msg_lines.append("\nThe output has to be a rectangular grid of numbers.\n")
        else:
            pred_for_display = pred_raw
            if pred_for_display.ndim < 2:
                pred_for_display = np.expand_dims(
                    pred_for_display, axis=list(range(2 - pred_for_display.ndim))
                )

            if pred_raw.shape != truth.shape:
                per_example_scores.append(0.0)
                msg_lines.append(
                    f"\n\nShape mismatch: your prediction's shape was {pred_raw.shape}, "
                    f"while the correct shape was {truth.shape}."
                )
            else:
                msg_lines.append(
                    "\nYour code's output does not match the expected output."
                    "\n\nBelow is a visualization of the 2D array your code produced as well as the expected output.\n"
                    "Correctly predicted values are shown as-is while the incorrectly predicted values are shown "
                    "in the format 'prediction/correct':\n"
                )
                diff = _array_diff(pred_for_display, truth)
                msg_lines.append(f"\n```\n{diff}\n```\n")

                example_score = float(np.mean(pred_raw == truth))
                example_score = float(
                    np.nan_to_num(example_score, posinf=0.0, neginf=0.0)
                )
                per_example_scores.append(example_score)
                msg_lines.append(
                    f"Output accuracy: {example_score:.2f} (0 is worst, 1 is best).\n"
                )

        if rr["error"]:
            msg_lines.append(
                f"\n\nYour code produced the following error:\n{rr['error']}\n"
            )

        feedback_parts.append("".join(msg_lines))

    full_feedback = "\n\n".join(feedback_parts)
    mean_score = (
        float(np.mean(np.nan_to_num(per_example_scores, posinf=0.0, neginf=0.0)))
        if per_example_scores
        else 0.0
    )
    return full_feedback, mean_score
