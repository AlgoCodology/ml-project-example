"""
Logging utilities
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str,
    config: dict,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup logger with consistent formatting
    
    Args:
        name: Logger name
        config: Configuration dictionary
        log_file: Optional log file name
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Get logging config
    log_level = config.get('logging', {}).get('level', 'INFO')
    log_format = config.get('logging', {}).get(
        'format',
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.setLevel(getattr(logging, log_level))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level))
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file or config.get('paths', {}).get('logs'):
        log_dir = Path(config.get('paths', {}).get('logs', 'logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_dir / log_file)
        file_handler.setLevel(getattr(logging, log_level))
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger