"""Turbovec MCP server entry point."""

from __future__ import annotations

import atexit
import os

from core.bootstrap import (
    fail_fast,
    quiet_stdio,
    setup_logging,
    silence_third_party_output,
)

logger = setup_logging()
logger.info("Initializing application and suppressing library outputs...")
silence_third_party_output()

try:
    from mcp.server.fastmcp import FastMCP
    import huggingface_hub.utils as hf_utils

    hf_utils.disable_progress_bars()

    # Model libraries print to stdout on import, which would corrupt MCP stdio.
    with quiet_stdio():
        from core.config import Settings
        from core.database import VectorDB
        from tools import register_tools_and_prompts
except Exception as exc:  # noqa: BLE001 - startup must report and stop
    fail_fast(logger, "Critical import error", exc)

LOCAL_HOST = "localhost"
LOCAL_PORT = 4392


def load_environment() -> None:
    """Loads a .env file when python-dotenv is available."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        logger.warning("python-dotenv not installed, continuing without .env file.")


def build_server(run_mode: str | None):
    """Creates the FastMCP server, the memory engine and registers the tools."""
    mcp_kwargs = {}
    if run_mode == "local":
        mcp_kwargs["host"] = LOCAL_HOST
        mcp_kwargs["port"] = LOCAL_PORT
    mcp = FastMCP("TurbovecSemanticSearch", **mcp_kwargs)

    logger.info("Initializing VectorDB...")
    with quiet_stdio():
        db = VectorDB(settings=Settings.from_env())

    logger.info("Registering tools and prompts...")
    register_tools_and_prompts(mcp, db)
    atexit.register(db.close)
    return mcp


def main() -> None:
    logger.info("Starting Turbovec MCP Server...")
    load_environment()
    run_mode = os.environ.get("run_mode")

    try:
        mcp = build_server(run_mode)
        logger.info("Turbovec MCP Server setup complete.")
    except Exception as exc:  # noqa: BLE001 - startup must report and stop
        fail_fast(logger, "Error during server initialization", exc)
        return

    if run_mode == "local":
        logger.info(
            f"Starting MCP server in local mode on http://{LOCAL_HOST}:{LOCAL_PORT}/sse"
        )
        mcp.run(transport="sse")
    else:
        logger.info("Starting MCP stdio transport...")
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
