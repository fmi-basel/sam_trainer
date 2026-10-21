import logging
from logging import DEBUG, INFO, WARNING, basicConfig

from rich.console import Console
from rich.logging import RichHandler


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.

    Args:
        name: Name for the logger (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def setup_logging(
    verbosity: int, console: Console, message_prefix: str = ""
) -> None:
    """Configure logging based on verbosity level.

    Args:
        verbosity: 0=WARNING, 1=INFO, 2=DEBUG
        console: Optional Rich console instance for logging
        message_prefix: Text prepended to every log message (e.g. a plate name to tell
            interleaved parallel workers apart)
    """
    level_map = {0: WARNING, 1: INFO, 2: DEBUG}
    level = level_map.get(verbosity, DEBUG)

    if console is None:
        console = Console()

    basicConfig(
        level=level,
        format=f"{message_prefix.replace('%', '%%')}%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
        force=True,
    )
