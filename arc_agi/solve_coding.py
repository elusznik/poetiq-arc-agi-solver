import re
from typing import Optional

import numpy as np

from arc_agi.interfaces import FeedbackGenerator, Problem, PromptGenerator, Sandbox
from arc_agi.llm import llm
from arc_agi.types import ARCAGIResult, ARCAGISolution, ExpertConfig, RunResult


async def solve_coding(
    *,
    problem: Problem,
    sandbox: Sandbox,
    feedback_generator: FeedbackGenerator,
    prompt_generator: PromptGenerator,
    config: ExpertConfig,
) -> ARCAGIResult:
    llm_model = config["llm_id"]
    max_iterations = int(config["max_iterations"])
    solver_temperature = float(config["solver_temperature"])
    # Unused config parameters in this refactor, but kept for reference/compatibility if needed later
    # max_solutions = config.get("max_solutions", 1)
    # shuffle_examples = config.get("shuffle_examples", False)
    # improving_order = config.get("improving_order", False)
    timeout_sandbox = float(config.get("timeout_s", 5))
    selection_probability = float(config.get("selection_probability", 1.0))
    seed = int(config.get("seed", 0))
    return_best = bool(config.get("return_best_result"))
    request_timeout = config.get("request_timeout")
    max_total_timeouts = config.get("max_total_timeouts")
    max_total_time = config.get("max_total_time")
    per_iteration_retries = config.get("per_iteration_retries")

    best_train_score = -1.0
    best_result: Optional[ARCAGIResult] = None
    last_train: list[RunResult] = [
        RunResult(
            success=False,
            output="",
            soft_score=0.0,
            error="Unexpected use of initial empty train result",
            code="",
        )
    ]
    last_test: Optional[list[RunResult]] = None

    rng = np.random.default_rng(seed)
    solutions: list[ARCAGISolution] = []

    for it in range(max_iterations):
        # Generate prompt using the generator
        # Note: shuffle logic should ideally be inside the generator or problem,
        # but for now we pass the seed implicitly via the loop or handle it in the adapter if needed.
        # The current ARCPromptGenerator implementation in the adapter doesn't use the seed yet,
        # but we can improve that later.
        message = prompt_generator.generate_solver_prompt(problem)

        selected = []
        if solutions:
            mask = rng.uniform(size=len(solutions)) < selection_probability
            selected = [s for s, keep in zip(solutions, mask, strict=False) if keep]

        if selected:
            message += prompt_generator.generate_feedback_prompt(selected)

        try:
            response, duration, max_total_time, max_total_timeouts = await llm(
                llm_model,
                message=message,
                temperature=solver_temperature,
                request_timeout=request_timeout,
                max_remaining_time=max_total_time,
                max_remaining_timeouts=max_total_timeouts,
                problem_id=getattr(problem, "problem_id", None),
                retries=per_iteration_retries,
            )
        except Exception as e:
            print(f"LLM call failed: {e}")
            if (
                "429" in str(e)
                or "Too Many Requests" in str(e)
                or "rate limit" in str(e).lower()
            ) or "Exceeded time allotted to the request" in str(e):
                print(
                    "Exiting early due to exceeding allotted time or timeouts on problem",
                    getattr(problem, "problem_id", "unknown"),
                )
                break
            continue

        print(f"DEBUG: LLM Response: {response[:100]}...")
        code = _parse_code_from_llm(response)
        print(f"DEBUG: Parsed Code: {code[:100] if code else 'None'}")
        if not code:
            continue

        # Execute on Train
        train_results_exec = []
        train_examples = problem.get_train_examples()
        # Assuming train_examples is list of (input, output) or similar.
        # For ARC, it is list of (input, output).

        # We need to map the generic ExecutionResult back to RunResult for internal tracking
        train_res_run_results = []

        for i, example in enumerate(train_examples):
            # Handle different example formats if needed, but for ARC it's (in, out)
            if isinstance(example, (tuple, list)) and len(example) == 2:
                inp, out = example
            else:
                inp = example  # Just input?

            exec_result = await sandbox.run(code, inp, timeout_s=timeout_sandbox)
            train_results_exec.append(exec_result)

            # Convert to RunResult for legacy compatibility
            # Note: soft_score calculation is now delegated to FeedbackGenerator if possible,
            # but we need it here for the loop logic (best_train_score).
            # The ARCFeedbackGenerator.generate_batch will handle this.
            train_res_run_results.append(
                RunResult(
                    success=exec_result.success,
                    output=exec_result.output,
                    soft_score=0.0,  # Will be filled by feedback generator
                    error=exec_result.error,
                    code=code,
                )
            )

        # Execute on Test
        test_results_exec = []
        test_examples = problem.get_test_examples()
        test_res_run_results = []

        for i, example in enumerate(test_examples):
            # Test examples might be just input
            inp = example
            if isinstance(example, dict) and "input" in example:
                inp = example["input"]

            exec_result = await sandbox.run(code, inp, timeout_s=timeout_sandbox)
            test_results_exec.append(exec_result)

            test_res_run_results.append(
                RunResult(
                    success=False,  # We don't know success on test usually
                    output=exec_result.output,
                    soft_score=0.0,
                    error=exec_result.error,
                    code=code,
                )
            )

        train_res = train_res_run_results
        test_res = test_res_run_results
        last_train, last_test = train_res, test_res

        # Generate Feedback
        # We use the batch generation method we added to the adapter
        # If the generator doesn't have generate_batch, we might need a fallback or update the interface.
        # For now, we assume we are using the ARC adapter which has it.
        if hasattr(feedback_generator, "generate_batch"):
            # We need to pass the raw train inputs/outputs to the generator?
            # The generator should probably know about the problem or we pass it.
            # ARCFeedbackGenerator.generate_batch(results, train_in, train_out)
            # We need to extract train_in/out from problem
            train_in = [ex[0] for ex in train_examples]
            train_out = [ex[1] for ex in train_examples]
            feedback, score = feedback_generator.generate_batch(
                train_results_exec, train_in, train_out
            )

            # Update soft scores in train_res based on what the generator calculated?
            # The generator returned a mean score.
            # Ideally the generator should update the results or return detailed scores.
            # For now, we trust the generator's score for the loop.
        else:
            feedback = "No feedback generator available."
            score = 0.0

        # Update success status based on score (ARC specific assumption: score 1.0 = success)
        if score == 1.0:
            # Mark all as success?
            for r in train_res:
                r["success"] = True

        if all(r["success"] for r in train_res):
            return ARCAGIResult(
                train_results=train_res, results=test_res, iteration=it + 1
            )

        solutions.append(ARCAGISolution(code=code, feedback=feedback, score=score))

        if score >= best_train_score:
            best_train_score = score
            best_result = ARCAGIResult(
                train_results=train_res, results=test_res, iteration=it + 1
            )

    if return_best and best_result is not None:
        return best_result
    if last_test is None:
        last_test = [
            RunResult(
                success=False,
                output="",
                soft_score=0.0,
                error="Failed to generate any valid solutions.",
                code="",
            )
        ]
    return ARCAGIResult(
        train_results=last_train, results=last_test, iteration=max_iterations
    )


def _parse_code_from_llm(response: str) -> Optional[str]:
    m = re.search(r"```python\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1)
    # Fallback: look for any code block
    m = re.search(r"```\s*(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1)
    # Fallback: assume the whole response is code if it contains typical python keywords
    if "print(" in response or "def " in response or "import " in response:
        return response
    return None
