"""Logging utility module"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from config import LOGGING_CONFIG

def setup_logger(name: str = "ara_bot", level: str = None) -> logging.Logger:
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name
        level: Logging level (defaults to config)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger  # Already configured
    
    level = level or LOGGING_CONFIG["LEVEL"]
    logger.setLevel(getattr(logging, level.upper()))
    
    # Formatter
    formatter = logging.Formatter(LOGGING_CONFIG["FORMAT"])
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    log_file = LOGGING_CONFIG["FILE"]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=LOGGING_CONFIG["MAX_BYTES"],
        backupCount=LOGGING_CONFIG["BACKUP_COUNT"],
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

