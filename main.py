import os
import sys
import warnings
import logging

def setup_logging():
    """Configure file-based logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("server.log", encoding="utf-8")
        ]
    )
    return logging.getLogger(__name__)

# Initialize logging before any imports or redirections
logger = setup_logging()
logger.info("Initializing application and suppressing library outputs...")

# 1. Disable warnings and progress bars via environment
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["DISABLE_TQDM"] = "1"

try:
    from mcp.server.fastmcp import FastMCP
    import huggingface_hub.utils as hf_utils
    hf_utils.disable_progress_bars()
    
    # Import our separated modules, suppressing their stdout/stderr output
    import contextlib
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            from vector_db import VectorDB
            from tools import register_tools_and_prompts
except Exception as e:
    logger.critical(f"Critical import error: {e}", exc_info=True)
    print(f"Critical import error: {e}", file=sys.stderr)
    sys.exit(1)

def main():
    logger.info("Starting Turbovec MCP Server...")

    try:
        # Initialize MCP Server
        mcp = FastMCP("TurbovecSemanticSearch")

        # Initialize Storage & Turbovec Index
        # This will also load the SentenceTransformer model
        logger.info("Initializing VectorDB...")
        import contextlib
        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                db = VectorDB()

        # Register tools
        logger.info("Registering tools and prompts...")
        register_tools_and_prompts(mcp, db)

        logger.info("Turbovec MCP Server setup complete.")
        print("Turbovec MCP Server is successfully running", file=sys.stderr)

    except Exception as e:
        logger.error(f"Error during server initialization: {e}", exc_info=True)
        print(f"Error during server initialization: {e}", file=sys.stderr)
        sys.exit(1)

    # Run the server using stdio transport
    logger.info("Starting MCP stdio transport...")
    mcp.run(transport='stdio')

if __name__ == "__main__":
    main()
