import pytest
from src.functions import add, divide


def test_add():
    result = add(1, 4)
    assert result == 5


def test_divide():
    with pytest.raises(ZeroDivisionError):
        divide(1, 0)
