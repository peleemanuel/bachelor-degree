import piexif
from typing import Optional
from PIL import Image

def _convert_to_deg(value) -> float:
    deg, min_, sec = value
    return deg[0]/deg[1] + (min_[0]/min_[1])/60 + (sec[0]/sec[1])/3600

class ExifGPS:
    def __init__(self, img_path: str):
        self._img_path = img_path
        self._gps = piexif.load(img_path).get("GPS", {})

    def path(self) -> str:
        """Return the path to the image file."""
        return self._img_path

    def latitude(self) -> Optional[float]:
        lat = self._gps.get(piexif.GPSIFD.GPSLatitude)
        ref = self._gps.get(piexif.GPSIFD.GPSLatitudeRef, b'N').decode()
        if not lat: return None
        deg = _convert_to_deg(lat)
        return -deg if ref == 'S' else deg

    def longitude(self) -> Optional[float]:
        lon = self._gps.get(piexif.GPSIFD.GPSLongitude)
        ref = self._gps.get(piexif.GPSIFD.GPSLongitudeRef, b'E').decode()
        if not lon: return None
        deg = _convert_to_deg(lon)
        return -deg if ref == 'W' else deg

    def altitude(self) -> Optional[float]:
        alt = self._gps.get(piexif.GPSIFD.GPSAltitude)
        ref = self._gps.get(piexif.GPSIFD.GPSAltitudeRef, 0)
        if not alt: return None
        m = alt[0]/alt[1]
        return -m if ref == 1 else m

    def img_direction(self) -> Optional[float]:
        dir_ = self._gps.get(piexif.GPSIFD.GPSImgDirection)
        if not dir_: return None
        return dir_[0]/dir_[1]
    
    def img_width(self) -> int:
        """Return the pixel width of this image."""
        with Image.open(self._img_path) as img:
            return img.width

    def img_height(self) -> int:
        """Return the pixel height of this image."""
        with Image.open(self._img_path) as img:
            return img.height