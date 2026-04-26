import os
import sys
import pytest
import tempfile
from src.utils import save_object, load_object


def test_save_and_load_object():
    obj = {"key": "value", "number": 42}
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_obj.pkl")
        save_object(path, obj)
        loaded = load_object(path)
        assert loaded == obj


def test_save_creates_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "nested", "dir", "obj.pkl")
        save_object(path, [1, 2, 3])
        assert os.path.exists(path)


def test_load_nonexistent_file_raises_exception():
    from src.exception import CustomException
    with pytest.raises(CustomException):
        load_object("nonexistent/path/file.pkl")
