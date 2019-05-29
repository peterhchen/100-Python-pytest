import mathlib
import pytest
import sys

# @pytest.mark.skipif (sys.version_info > (3,5), reason="Skip this test")
def test_calc_total():
    total = mathlib.calc_total (4, 5)
    assert total == 9

def test_calc_multiply ():
    result = mathlib.calc_multiply (10, 3)
    assert result == 30

def test_dummy():
    assert True

@pytest.mark.windows
def test_windows_1 ():
    assert True

@pytest.mark.windows
def test_windows_2 ():
    assert True

@pytest.mark.mac
def test_mac_1 ():
        assert True

@pytest.mark.mac
def test_mac_2 ():
    assert True