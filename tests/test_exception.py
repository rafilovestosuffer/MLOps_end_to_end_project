import sys
import pytest
from src.exception import CustomException


def test_custom_exception_is_raised():
    with pytest.raises(CustomException):
        try:
            x = 1 / 0
        except Exception as e:
            raise CustomException(e, sys)


def test_custom_exception_message_contains_file():
    try:
        try:
            x = 1 / 0
        except Exception as e:
            raise CustomException(e, sys)
    except CustomException as ce:
        assert "File" in str(ce)


def test_custom_exception_message_contains_line_number():
    try:
        try:
            x = 1 / 0
        except Exception as e:
            raise CustomException(e, sys)
    except CustomException as ce:
        assert "Line number" in str(ce)


def test_custom_exception_message_contains_error():
    try:
        try:
            x = 1 / 0
        except Exception as e:
            raise CustomException(e, sys)
    except CustomException as ce:
        assert "division by zero" in str(ce)
