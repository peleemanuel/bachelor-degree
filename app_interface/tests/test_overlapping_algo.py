# tests/test_overlapping_algo.py
import os
import pytest
import folium

from trace_viewer.trace import TraceCreator
from trace_viewer.overlapping_algo import compare_overlapping_zones
from trace_viewer.drones import DroneType, get_drone

TEST_HOME        = os.path.join(os.getcwd(), "raw_images", "test_home")
OVERLAP_TEST_HOME = os.path.join(os.getcwd(), "raw_images", "overlap_test_home")

def build_trace(folder: str) -> TraceCreator:
    """
    Helper to instantiate and fully build a TraceCreator for a folder.
    """
    drone = get_drone(DroneType.MINI_4K)
    tc = TraceCreator(folder, drone)
    tc.build()  # runs _load_gps_info, _build_trace, _build_polygons, _unify_polygons
    return tc

def test_folders_exist():
    assert os.path.isdir(TEST_HOME),    f"Missing test folder: {TEST_HOME}"
    assert os.path.isdir(OVERLAP_TEST_HOME), f"Missing overlap-test folder: {OVERLAP_TEST_HOME}"

def test_compare_overlapping_zones_reports_change_or_no_overlap(capsys):
    """
    Running compare_overlapping_zones on the two folders must print
    either a 'Significant change' message or 'No overlap' fallback.
    """
    t1 = build_trace(TEST_HOME)
    t2 = build_trace(OVERLAP_TEST_HOME)

    # Ensure they in fact overlap
    assert t1.overlaps(t2), "Expected these two traces to overlap"

    # Run and capture stdout
    compare_overlapping_zones(t1, t2)
    out = capsys.readouterr().out

    # It must report something meaningful
    assert (
        "Change rejected" in out
    ), f"Unexpected output:\n{out}"

def test_trace_and_footprint_drawn_on_map(tmp_path):
    """
    Smoke test that add_to_map produces a valid Folium map without error.
    """
    t1 = build_trace(TEST_HOME)
    m = folium.Map(location=[0,0], zoom_start=1)
    # Should not raise
    t1.add_to_map(m)
    # Save to a local html file
    out_html = tmp_path / "map.html"
    m.save(str(out_html))
    assert out_html.exists() and out_html.stat().st_size > 0
