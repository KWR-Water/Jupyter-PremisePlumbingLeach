# Python

import pytest
from ImportFunctions import compute_fraction_exceedance
import numpy as np

from ImportFunctions import sec_to_days_hours_min_sec
from ImportFunctions import days_hours_min_sec_to_sec

def test_sec_to_days_hours_min_sec_zero():
    assert sec_to_days_hours_min_sec(0) == (0, 0, 0, 0)

def test_sec_to_days_hours_min_sec_one_second():
    assert sec_to_days_hours_min_sec(1) == (0, 0, 0, 1)

def test_sec_to_days_hours_min_sec_one_minute():
    assert sec_to_days_hours_min_sec(60) == (0, 0, 1, 0)

def test_sec_to_days_hours_min_sec_one_hour():
    assert sec_to_days_hours_min_sec(3600) == (0, 1, 0, 0)

def test_sec_to_days_hours_min_sec_one_day():
    assert sec_to_days_hours_min_sec(86400) == (1, 0, 0, 0)

def test_sec_to_days_hours_min_sec_complex():
    assert sec_to_days_hours_min_sec(90061) == (1, 1, 1, 1)

def test_sec_to_days_hours_min_sec_large():
    assert sec_to_days_hours_min_sec(172801) == (2, 0, 0, 1)

def test_sec_to_days_hours_min_sec_float():
    assert sec_to_days_hours_min_sec(3661.5) == (0, 1, 1, 1)

def test_days_hours_min_sec_to_sec_zero():
    assert days_hours_min_sec_to_sec(0, 0, 0, 0) == 0

def test_days_hours_min_sec_to_sec_one_second():
    assert days_hours_min_sec_to_sec(0, 0, 0, 1) == 1

def test_days_hours_min_sec_to_sec_one_minute():
    assert days_hours_min_sec_to_sec(0, 0, 1, 0) == 60

def test_days_hours_min_sec_to_sec_one_hour():
    assert days_hours_min_sec_to_sec(0, 1, 0, 0) == 3600

def test_days_hours_min_sec_to_sec_one_day():
    assert days_hours_min_sec_to_sec(1, 0, 0, 0) == 86400

def test_days_hours_min_sec_to_sec_complex():
    assert days_hours_min_sec_to_sec(1, 1, 1, 1) == 90061

def test_days_hours_min_sec_to_sec_large():
    assert days_hours_min_sec_to_sec(2, 0, 0, 1) == 172801

def test_days_hours_min_sec_to_sec_float():
    assert days_hours_min_sec_to_sec(0, 1, 1, 1) == 3661



def test_compute_fraction_exceedance_simple():
    demand = np.array([[[1, 2], [3, 4]]])  # shape: (1, 2, 2)
    water_quality = np.array([[[0.5, 1.5], [2.5, 3.5]]])  # shape: (1, 2, 2)
    threshold = 2.0
    # Only values 2.5, 3.5, and 3.0 (from demand) exceed threshold
    result = compute_fraction_exceedance(demand, water_quality, threshold)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1,)
    assert 0.0 <= result[0] <= 1.0

def test_compute_fraction_exceedance_all_below():
    demand = np.ones((2, 3, 4)) *0.1
    water_quality = np.ones((2, 3, 4)) *0.1
    threshold = 1.0
    result = compute_fraction_exceedance(demand, water_quality, threshold)
    assert np.all(result == 0)

def test_compute_fraction_exceedance_all_above():
    demand = np.ones((2, 3, 4)) * 10
    water_quality = np.ones((2, 3, 4)) * 10
    threshold = 1.0
    result = compute_fraction_exceedance(demand, water_quality, threshold)
    assert np.all(result == 1)

def test_compute_fraction_exceedance_mixed():
    demand = np.array([[[0, 2], [4, 6]]])
    water_quality = np.array([[[1, 3], [5, 7]]])
    threshold = 4
    result = compute_fraction_exceedance(demand, water_quality, threshold)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1,)
    assert 0.0 <= result[0] <= 1.0
