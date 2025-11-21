from arc_agi.arc_adapter import (
    ARCFeedbackGenerator,
    ARCProblem,
    ARCPromptGenerator,
    ARCSandbox,
)
from arc_agi.config import CONFIG_LIST
from arc_agi.prompts import FEEDBACK_PROMPT, SOLVER_PROMPT_1
from arc_agi.solve_parallel_coding import solve_parallel_coding
from arc_agi.types import ARCAGIResult


async def solve(
    train_in: list[list[list[int]]],
    train_out: list[list[list[int]]],
    test_in: list[list[list[int]]],
    problem_id: str | None = None,
) -> list[ARCAGIResult]:
    problem = ARCProblem(train_in, train_out, test_in, problem_id)
    sandbox = ARCMCPSandbox()
    # Use prompts from the first config or constants.
    # Note: Different experts might use different prompts in the original code,
    # but here we are standardizing for the refactor.
    # If experts need different prompts, PromptGenerator should handle it or we need multiple generators.
    # For now, we assume uniform prompts for simplicity or use the one from config.

    # Actually, expert_configs might have different prompts.
    # The solver loop uses prompt_generator.
    # If we want to support per-expert prompts, we might need to pass the generator factory or
    # have the generator look at the config?
    # But solve_coding takes `prompt_generator` AND `config`.
    # In my refactor of solve_coding, I used `prompt_generator.generate_solver_prompt(problem)`.
    # I ignored `config['solver_prompt']`.

    # To maintain exact behavior, ARCPromptGenerator should probably accept the config?
    # Or we create a new generator for each expert in solve_parallel_coding?
    # solve_parallel_coding receives a SINGLE prompt_generator.

    # Let's stick to a single generator for now using the default prompts.
    # Create a prompt generator for each expert config to support different prompts
    prompt_generators = [
        ARCPromptGenerator(
            cfg.get("solver_prompt", SOLVER_PROMPT_1),
            cfg.get("feedback_prompt", FEEDBACK_PROMPT),
        )
        for cfg in CONFIG_LIST
    ]
    feedback_generator = ARCFeedbackGenerator()

    result = await solve_parallel_coding(
        problem=problem,
        sandbox=sandbox,
        feedback_generator=feedback_generator,
        prompt_generator=prompt_generators,
        expert_configs=[cfg.copy() for cfg in CONFIG_LIST],
    )

    return result
