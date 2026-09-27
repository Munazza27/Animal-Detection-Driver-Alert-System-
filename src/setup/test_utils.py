"""
test_utils.py
-------------
Basic sanity tests for the geometry helper functions in src/utils.py.
Run with: python -m pytest tests/test_utils.py -v
(or just: python tests/test_utils.py)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import box_center_x, box_area_fraction, get_horizontal_zone


def test_box_center_x():
    assert box_center_x(0, 100) == 50
    assert box_center_x(200, 400) == 300


def test_box_area_fraction():
    # box of 100x100 inside a 1000x1000 frame = 1% of frame area
    frac = box_area_fraction(0, 0, 100, 100, 1000, 1000)
    assert abs(frac - 0.01) < 1e-6


def test_get_horizontal_zone_left():
    # center at 10% of a 1000px wide frame -> LEFT (below 0.35 boundary)
    zone = get_horizontal_zone(100, 1000, left_end=0.35, right_start=0.65)
    assert zone == "LEFT"


def test_get_horizontal_zone_center():
    zone = get_horizontal_zone(500, 1000, left_end=0.35, right_start=0.65)
    assert zone == "CENTER"


def test_get_horizontal_zone_right():
    zone = get_horizontal_zone(900, 1000, left_end=0.35, right_start=0.65)
    assert zone == "RIGHT"


if __name__ == "__main__":
    test_box_center_x()
    test_box_area_fraction()
    test_get_horizontal_zone_left()
    test_get_horizontal_zone_center()
    test_get_horizontal_zone_right()
    print("All tests passed.")