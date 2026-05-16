"""Centralized logging with loguru + Rich integration."""

import sys
from pathlib import Path
from loguru import logger

from config import get

_LOG_CONFIGURED = False


def setup_logger():
    """Configure loguru with file rotation and Rich-compatible format."""
    global _LOG_CONFIGURED
    if _LOG_CONFIGURED:
        return logger
    _LOG_CONFIGURED = True

    logger.remove()

    level = get("logging.level", "INFO")
    log_file = get("logging.file", "logs/trader.log")
    rotation = get("logging.rotation", "10 MB")
    retention = get("logging.retention", "30 days")

    # Console — compact format
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )

    # File — full format with rotation
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_path),
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <7} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation=rotation,
        retention=retention,
        compression="gz",
        encoding="utf-8",
        catch=True,           # suppress PermissionError on rotation (Windows lock)
    )

    return logger


log = setup_logger()
