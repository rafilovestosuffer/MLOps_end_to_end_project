import os
import sys
import pytest

def test_logger_creates_log_file():
    import src.logger
    log_base = os.path.join(os.getcwd(), "logs")
    assert os.path.exists(log_base), "logs/ directory should be created by logger"

def test_logger_file_has_correct_extension():
    from src.logger import LOG_FILE
    assert LOG_FILE.endswith(".log"), "Log file should have .log extension"

def test_logger_file_path_exists():
    from src.logger import LOG_FILE_PATH
    assert LOG_FILE_PATH.endswith(".log"), "LOG_FILE_PATH should point to a .log file"
