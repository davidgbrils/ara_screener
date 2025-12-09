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
    
    # Console handler with safe encoding for Windows compatibility
    class SafeStreamHandler(logging.StreamHandler):
        """StreamHandler that safely handles encoding errors on Windows"""
        def emit(self, record):
            try:
                msg = self.format(record)
                stream = self.stream
                # Try to write normally first
                try:
                    stream.write(msg + self.terminator)
                    self.flush()
                except (UnicodeEncodeError, ValueError) as e:
                    # If encoding error, remove problematic characters
                    try:
                        # Remove non-ASCII characters that cause issues
                        msg_clean = msg.encode('ascii', errors='ignore').decode('ascii')
                        stream.write(msg_clean + self.terminator)
                        self.flush()
                    except Exception:
                        # If still fails, try to write to file handler only
                        pass
            except Exception:
                # Silently ignore if stream is closed or other errors
                pass
    
    console_handler = SafeStreamHandler(sys.stdout)
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

