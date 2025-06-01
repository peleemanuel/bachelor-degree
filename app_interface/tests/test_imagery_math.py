# tests/test_imagery_math.py

import math
import pytest
from trace_viewer.imagery_math import (
    calculate_gsd,
    latlon_to_pixel,
    image_corners,
    METERS_PER_DEGREE,
)

def test_calculate_gsd_simple():
    # altitude=100m, sensor=10mm, focal=5mm, width=1000px => 20 cm/pix
    gsd = calculate_gsd(100.0, 10.0, 5.0, 1000)
    assert gsd == pytest.approx(20.0)

def test_latlon_to_pixel_center():
    px, py = latlon_to_pixel(
        lat=0.0, lon=0.0,
        center_lat=0.0, center_lon=0.0,
        direction_deg=0.0,
        gsd_cm=10.0,
        img_width=500, img_height=300
    )
    assert px == pytest.approx(250.0)
    assert py == pytest.approx(150.0)

def test_latlon_to_pixel_offset():
    # 1 meter east → 10 px to the right (with gsd=10cm/pix)
    offset_deg = 1 / METERS_PER_DEGREE
    px, py = latlon_to_pixel(
        lat=0.0, lon=offset_deg,
        center_lat=0.0, center_lon=0.0,
        direction_deg=0.0,
        gsd_cm=10.0,
        img_width=200, img_height=200
    )
    assert px == pytest.approx(100 + 10)
    assert py == pytest.approx(100)

def test_image_corners_output():
    # Just ensure four corners returned, not center
    corners = image_corners(
        center_lat=0.0, center_lon=0.0,
        direction_deg=45.0,
        gsd_cm=1.0,
        img_width=10, img_height=10,
        adjust_for_mini4k=False
    )
    assert isinstance(corners, list) and len(corners) == 4
    for lat, lon in corners:
        assert not (lat == pytest.approx(0.0) and lon == pytest.approx(0.0))
