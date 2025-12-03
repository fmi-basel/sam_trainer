from logging import DEBUG, INFO, WARNING, basicConfig

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(verbosity: int) -> None:
    """Configure logging based on verbosity level."""
    level_map = {0: WARNING, 1: INFO, 2: DEBUG}
    level = level_map.get(verbosity, DEBUG)

    basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=Console(), rich_tracebacks=True)],
    )
