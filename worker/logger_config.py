import logging
from typing import Union


def setup_logger(logger_name: str, level: Union[int, str] = logging.DEBUG):
    """Create a logger and accept either integer or string levels.

    Accepts standard logging level ints or string names like 'DEBUG'.
    """
    logger = logging.getLogger(logger_name)

    # Allow passing string level names (e.g. 'DEBUG') or ints
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())

    logger.setLevel(level)

    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger