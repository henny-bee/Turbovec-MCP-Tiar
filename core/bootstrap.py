"""Process bootstrap: logging and third-party output suppression.

The MCP stdio transport uses stdout as its wire protocol, so anything a
library prints there corrupts the session. Everything noisy is silenced before
the heavy imports happen, and diagnostics go to a log file instead.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import warnings
from typing import Iterator

__all__ = ["setup_logging", "silence_third_party_output", "quiet_stdio"]

# Environment switches understood by transformers / huggingface / tqdm.
_QUIET_ENV = {
    "TRANSFORMERS_VERBOSITY": "error",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "TQDM_DISABLE": "1",
    "DISABLE_TQDM": "1",
}


def setup_logging(log_file: str = "server.log", level: int = logging.INFO):
    """Configures file-based logging and returns the root application logger."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8")],
    )
    return logging.getLogger("turbovec")


def silence_third_party_output() -> None:
    """Disables warnings and progress bars before model libraries are imported."""
    warnings.filterwarnings("ignore")
    for key, value in _QUIET_ENV.items():
        os.environ.setdefault(key, value)


@contextlib.contextmanager
def quiet_stdio() -> Iterator[None]:
    """Redirects stdout/stderr to the null device for the duration of a block."""
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def fail_fast(logger: logging.Logger, message: str, exc: BaseException) -> "None":
    """Logs a fatal startup error and exits with a non-zero status."""
    logger.critical(f"{message}: {exc}", exc_info=True)
    sys.exit(1)
