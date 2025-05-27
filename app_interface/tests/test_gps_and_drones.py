import pytest
import piexif
from trace_viewer.drones import DroneRegistry, DroneSpec, DroneType
from trace_viewer.gps import GPSInfo

# Test DroneRegistry functionality
def test_drone_registry_contains_builtins():
    # Ensure built-in drones are registered
    all_types = DroneRegistry._specs.keys()
    assert DroneType.SELF_MADE in all_types
    assert DroneType.GENERIC_DJI in all_types
    assert DroneType.MINI_4K in all_types

@ pytest.mark.parametrize("dtype,sensor,focal", [
    (DroneType.SELF_MADE, 5.02, 6),
    (DroneType.GENERIC_DJI, 6.3, 4.5),
    (DroneType.MINI_4K, pytest.approx(6.3/4), 4.5),
])
def test_drone_specs(dtype, sensor, focal):
    spec = DroneRegistry.get(dtype)
    assert isinstance(spec, DroneSpec)
    assert spec.type == dtype
    assert spec.sensor_width_mm == pytest.approx(sensor)
    assert spec.focal_length_mm == pytest.approx(focal)

# Test GPSInfo EXIF parsing
def make_test_exif(tmp_path, lat_vals, lat_ref, lon_vals, lon_ref, alt_val=None, alt_ref=0, img_dir=None):
    """
    Create a minimal JPEG file with GPS EXIF tags for testing.
    """
    # Create a blank JPEG
    img_path = tmp_path / "test.jpg"
    with open(img_path, "wb") as f:
        f.write(b"\xff\xd8\xff\xd9")  # SOI + EOI

    # Build EXIF
    gps_ifd = {}
    # Helper to rational
    def rat(x): return (int(x), 1)

    gps_ifd[piexif.GPSIFD.GPSLatitude] = [(lat_vals[0],1),(lat_vals[1],1),(lat_vals[2],1)]
    gps_ifd[piexif.GPSIFD.GPSLatitudeRef] = lat_ref.encode()
    gps_ifd[piexif.GPSIFD.GPSLongitude] = [(lon_vals[0],1),(lon_vals[1],1),(lon_vals[2],1)]
    gps_ifd[piexif.GPSIFD.GPSLongitudeRef] = lon_ref.encode()
    if alt_val is not None:
        gps_ifd[piexif.GPSIFD.GPSAltitude] = rat(alt_val)
        gps_ifd[piexif.GPSIFD.GPSAltitudeRef] = alt_ref

    exif = {"GPS": gps_ifd}
    exif_bytes = piexif.dump(exif)
    piexif.insert(exif_bytes, str(img_path))
    return str(img_path)


def test_gpsinfo_latlonalt_direction(tmp_path):
    # Create EXIF with known values: 10°20'30" N, 40°50'10" W, alt=100m
    img = make_test_exif(tmp_path, lat_vals=(10,20,30), lat_ref='N', lon_vals=(40,50,10), lon_ref='W', alt_val=100, alt_ref=0)
    info = GPSInfo(img)
    lat = info.latitude()
    lon = info.longitude()
    alt = info.altitude()
    # Convert d/m/s
    expected_lat = 10 + 20/60 + 30/3600
    expected_lon = -(40 + 50/60 + 10/3600)
    assert lat == pytest.approx(expected_lat)
    assert lon == pytest.approx(expected_lon)
    assert alt == pytest.approx(100)

    # No direction tag -> None
    assert info.img_direction() is None

# Edge cases: missing tags
@ pytest.mark.parametrize("tag,attr", [
    ("GPSLatitude", lambda i: i.latitude()),
    ("GPSLongitude", lambda i: i.longitude()),
    ("GPSAltitude", lambda i: i.altitude()),
])
def test_missing_tags(tmp_path, tag, attr):
    # Create empty JPEG
    img_path = tmp_path / "no_gps.jpg"
    with open(img_path, "wb") as f: f.write(b"\xff\xd8\xff\xd9")
    info = GPSInfo(str(img_path))
    assert attr(info) is None
