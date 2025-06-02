import pytest
import piexif
from trace_viewer.drones import DroneRegistry, DroneSpec, DroneType
from trace_viewer.gps import ExifGPS
from PIL import Image

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

# Test ExifGPS EXIF parsing
def make_test_exif(tmp_path, lat_vals, lat_ref, lon_vals, lon_ref,
                   alt_val=None, alt_ref=0):
    """
    Create a minimal but valid JPEG file with GPS EXIF tags for testing.
    """
    # Create a small RGB image and save it as JPEG
    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (10,10), (255,255,255)).save(str(img_path), format="JPEG")

    # Build the GPS IFD dict
    gps_ifd = {}
    # Rational helper
    def rat(x): return (int(x), 1)

    gps_ifd[piexif.GPSIFD.GPSLatitude] = [rat(v) for v in lat_vals]
    gps_ifd[piexif.GPSIFD.GPSLatitudeRef] = lat_ref.encode()
    gps_ifd[piexif.GPSIFD.GPSLongitude] = [rat(v) for v in lon_vals]
    gps_ifd[piexif.GPSIFD.GPSLongitudeRef] = lon_ref.encode()
    if alt_val is not None:
        gps_ifd[piexif.GPSIFD.GPSAltitude] = rat(alt_val)
        gps_ifd[piexif.GPSIFD.GPSAltitudeRef] = alt_ref

    # Dump & insert EXIF back into that file
    exif_dict = {"GPS": gps_ifd}
    exif_bytes = piexif.dump(exif_dict)
    piexif.insert(exif_bytes, str(img_path))

    return str(img_path)

def test_ExifGPS_latlonalt_direction(tmp_path):
    img = make_test_exif(tmp_path, lat_vals=(10,20,30), lat_ref='N', lon_vals=(40,50,10), lon_ref='W', alt_val=100, alt_ref=0)
    info = ExifGPS(img)
    lat = info.latitude()
    lon = info.longitude()
    alt = info.altitude()
    # Convert d/m/s
    expected_lat = 10 + 20/60 + 30/3600
    expected_lon = -(40 + 50/60 + 10/3600)
    assert lat == pytest.approx(expected_lat)
    assert lon == pytest.approx(expected_lon)
    assert alt == pytest.approx(100)

    assert info.img_direction() is None

@ pytest.mark.parametrize("tag,attr", [
    ("GPSLatitude", lambda i: i.latitude()),
    ("GPSLongitude", lambda i: i.longitude()),
    ("GPSAltitude", lambda i: i.altitude()),
])
def test_missing_tags(tmp_path, tag, attr):
    # Create empty JPEG
    img_path = tmp_path / "no_gps.jpg"
    with open(img_path, "wb") as f: f.write(b"\xff\xd8\xff\xd9")
    info = ExifGPS(str(img_path))
    assert attr(info) is None
