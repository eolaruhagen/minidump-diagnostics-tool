"""
Centralized logging module for the minidump diagnostics tool.
Provides a consistent logging interface across all modules.
"""
import logging
import sys
from typing import Optional
from pathlib import Path

class AppLogger:    
    _instance: Optional['AppLogger'] = None
    _logger: Optional[logging.Logger] = None
    
    def __new__(cls) -> 'AppLogger':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._logger is None:
            self._setup_logger()
    
    def _setup_logger(self) -> None:
        self._logger = logging.getLogger('minidump_diagnostics')
        self._logger.setLevel(logging.INFO)
        
        if not self._logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            
            self._logger.addHandler(console_handler)
    
    def info(self, message: str) -> None:
        self._logger.info(message)
    
    def warning(self, message: str) -> None:
        self._logger.warning(message)
    
    def error(self, message: str) -> None:
        self._logger.error(message)
    
    def debug(self, message: str) -> None:
        self._logger.debug(message)
    
    def critical(self, message: str) -> None:
        self._logger.critical(message)


logger = AppLogger()
