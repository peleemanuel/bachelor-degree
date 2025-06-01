# tests/test_imagery_io.py

import pytest
import folium
from PIL import Image
from trace_viewer.imagery_io import load_image, add_corners_to_map

def test_load_image(tmp_path):
    img_path = tmp_path / "img.png"
    Image.new("RGB", (8,8), (10,20,30)).save(str(img_path))
    img = load_image(str(img_path))
    assert isinstance(img, Image.Image)
    assert img.size == (8,8)

def test_add_corners_to_map_creates_polygon():
    m = folium.Map(location=[0,0], zoom_start=2)
    corners = [(0,0),(0,1),(1,1),(1,0)]
    add_corners_to_map(m, corners, color="green")
    # Folium's children dict should include at least one Polygon
    names = [type(obj).__name__ for obj in m._children.values()]
    assert "Polygon" in names
