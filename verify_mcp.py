import asyncio
import os
import sys

# Add current directory to path so we can import arc_agi
sys.path.append(os.getcwd())

from arc_agi.sandbox import MCPSandbox


async def main():
    print("Initializing MCPSandbox...")
    # Uses MCP_SERVER_COMMAND and MCP_SERVER_ARGS env vars if set, else defaults to uvx
    sandbox = MCPSandbox()

    code = "print('Hello from MCP!')"
    print(f"Running code: {code}")

    try:
        result = await sandbox.run(code, input_data=None)
        print("Result:", result)
        if result.success and "Hello from MCP!" in result.output:
            print("SUCCESS: MCPSandbox verification passed.")
        else:
            print("FAILURE: MCPSandbox verification failed.")
            print("\nIf you are using a local MCP server, please set:")
            print("export MCP_SERVER_COMMAND='/path/to/server'")
            print("export MCP_SERVER_ARGS='arg1 arg2'")
    except Exception as e:
        print(f"ERROR: {e}")
        print("\nIf you are using a local MCP server, please set:")
        print("export MCP_SERVER_COMMAND='/path/to/server'")
        print("export MCP_SERVER_ARGS='arg1 arg2'")


if __name__ == "__main__":
    asyncio.run(main())
