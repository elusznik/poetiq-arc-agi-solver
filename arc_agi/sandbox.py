import asyncio
import json
import os
import sys
import tempfile
import textwrap


async def run(
    code: str, input_grid: list[list[int]], timeout_s: float = 1.5
) -> tuple[bool, str]:
    """Run user code in a subprocess asynchronously, returning (ok, result or error)."""
    script = _build_script(code)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "u.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(script))

        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=td,
            env={"PYTHONHASHSEED": "0"},
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=json.dumps({"input": input_grid}).encode()),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return False, "timeout"

        if proc.returncode != 0:
            return False, (stderr.decode() or stdout.decode()).strip()

        try:
            payload = json.loads(stdout.decode())
            return bool(payload.get("ok")), json.dumps(payload.get("result"))
        except Exception as e:
            return False, f"bad-json: {e}"


def _build_script(code: str) -> str:
    return f"""
# generated file
{code}
if __name__ == '__main__':
    import json
    import numpy as np
    import scipy
    from sys import stdin
    data = json.load(stdin)
    res = transform(np.array(data['input']))
    print(json.dumps({{"ok": True, 'result': res.tolist()}}))
"""


from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from arc_agi.interfaces import ExecutionResult, Sandbox


class MCPSandbox(Sandbox):
    def __init__(self, command: str | None = None, args: list[str] | None = None):
        import os
        import shlex

        env_command = os.getenv("MCP_SERVER_COMMAND")
        env_args = os.getenv("MCP_SERVER_ARGS")

        # Default to the user-provided command if no env var or arg is set
        self.command = command or env_command or "uvx"

        if args:
            self.args = args
        elif env_args:
            self.args = shlex.split(env_args)
        else:
            # Default args for the user's specific server
            self.args = [
                "--from",
                "git+https://github.com/elusznik/mcp-server-code-execution-mode",
                "mcp-server-code-execution-mode",
                "run",
            ]

    async def run(
        self, code: str, input_data: Any, timeout_s: float = 1.5
    ) -> ExecutionResult:
        """Execute code using the MCP server."""

        # Prepare the code wrapper similar to legacy sandbox but for MCP
        # The MCP server's run_python tool expects a standalone script.
        # We need to wrap the user code to handle input/output if the user code expects it.
        # However, for the math example, it just prints.
        # For ARC, it expects a transform function.

        # Let's assume the input code is complete or we wrap it if needed.
        # For ARC compatibility, we might need to wrap it like _build_script does.
        # But MCPSandbox should be generic.
        # Let's assume the caller handles wrapping if specific input/output is needed,
        # OR we detect if it's an ARC problem?
        # The Sandbox interface takes `input_data`.
        # If `input_data` is provided, we should probably inject it or pass it.

        # For generic usage, we can inject input_data as a global variable or via stdin if the tool supports it.
        # The mcp-server-code-execution-mode run_python tool usually just runs the code.
        # It doesn't explicitly support passing stdin in the tool arguments (usually).
        # Let's check the tool definition if possible, but for now assume we need to embed data.

        full_code = code
        if input_data is not None:
            # Embed input data as a JSON string variable if needed
            # For now, we just pass the code as is, assuming the user handles it
            # or we might need to inject it.
            # data_json = json.dumps(input_data)
            pass

        # For ARC specifically, the `code` passed to `run` is just the `transform` function.
        # The `ARCSandbox` calls `_build_script` which adds the harness.
        # If we use `MCPSandbox` for ARC, we need that harness.
        # But `MCPSandbox` shouldn't know about ARC.
        # So `ARCProblem` or `solve.py` should prepare the full script?
        # Currently `ARCSandbox.run` calls `_build_script`.
        # If we want to replace `ARCSandbox` with `MCPSandbox` for ARC, we need an adapter or
        # `MCPSandbox` needs to be subclassed for ARC?
        # Or we make `ARCSandbox` use `MCPSandbox` internally?

        # Let's implement `MCPSandbox` as a raw code executor.
        # And for ARC, we might need `ARCMCPSandbox` that wraps the code.

        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
        )

        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # Call the run_python tool
                    # We assume the tool is named "run_python" and takes "code" argument.
                    result = await session.call_tool(
                        "run_python", arguments={"code": full_code}
                    )

                    # Parse result
                    # The tool returns a list of content blocks (TextContent or ImageContent)
                    # We expect TextContent.
                    output_text = ""
                    error_text = None
                    success = True  # Assume success unless we detect error?
                    # The tool might return error in text or raise exception?

                    for content in result.content:
                        if content.type == "text":
                            output_text += content.text

                    if result.isError:
                        success = False
                        error_text = output_text

                    return ExecutionResult(
                        success=success, output=output_text, error=error_text
                    )

        except Exception as e:
            return ExecutionResult(success=False, output="", error=str(e))
