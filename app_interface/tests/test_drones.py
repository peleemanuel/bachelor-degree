# tests/test_drones.py

import pytest
from trace_viewer.drones import DroneRegistry, DroneSpec, DroneType

def test_builtin_registry():
    # Registry should have exactly these keys
    keys = set(DroneRegistry._specs.keys())
    expected = {DroneType.SELF_MADE, DroneType.GENERIC_DJI, DroneType.MINI_4K}
    assert keys == expected

@pytest.mark.parametrize("dtype,sw,fl", [
    (DroneType.SELF_MADE, 5.02, 6.0),
    (DroneType.GENERIC_DJI, 6.3, 4.5),
    (DroneType.MINI_4K, pytest.approx(6.3/4), 4.5),
])
def test_drone_spec_values(dtype, sw, fl):
    spec = DroneRegistry.get(dtype)
    assert isinstance(spec, DroneSpec)
    assert spec.type == dtype
    assert spec.sensor_width_mm == pytest.approx(sw)
    assert spec.focal_length_mm == pytest.approx(fl)
