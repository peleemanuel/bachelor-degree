#!/usr/bin/env python3
import sys
import piexif
from enum import Enum
from pathlib import Path

class DroneType(Enum):
    SELF_MADE_DRONE = "selfMadeDrone"
    DJI = "DJI"

class GPSInfo():
    """
    Class that will save the GPS information from an image and its path
    """
    def __init__(self, path: 'Path', drone_type: DroneType, secondary_file: str = None):
        self.path = Path(path)
        self.lat = None
        self.lon = None
        self.alt = None
        self.secondary_file = secondary_file
        self.drone_type = drone_type

    def __str__(self):
        return f"GPSInfo(path={self.path}, lat={self.lat}, lon={self.lon}, alt={self.alt})"
      
    def convert_to_degrees(self, value):
        d = value[0][0] / value[0][1]
        m = value[1][0] / value[1][1]
        s = value[2][0] / value[2][1]
        return d + (m / 60.0) + (s / 3600.0)

    def extract_gps_info(self):
        if self.drone_type == DroneType.SELF_MADE_DRONE:
            return self.extract_gps_info_from_self_made_drone()
        elif self.drone_type == DroneType.DJI:
            return self.extract_gps_info_from_dji()
        else:
            raise ValueError("Unknown drone type")
    
    def extract_gps_info_from_self_made_drone(self):
        #TODO: Implement the logic to extract GPS info from self-made drone
        pass

    def extract_gps_info_from_dji(self):
        exif_dict = piexif.load(self.path)
        gps_ifd = exif_dict.get("GPS", {})

        if not gps_ifd:
            return None

        lat_ref = gps_ifd.get(piexif.GPSIFD.GPSLatitudeRef, b'').decode()
        lon_ref = gps_ifd.get(piexif.GPSIFD.GPSLongitudeRef, b'').decode()

        lat = gps_ifd.get(piexif.GPSIFD.GPSLatitude, None)
        lon = gps_ifd.get(piexif.GPSIFD.GPSLongitude, None)
        alt = gps_ifd.get(piexif.GPSIFD.GPSAltitude, None)
        alt_ref = gps_ifd.get(piexif.GPSIFD.GPSAltitudeRef, 0)

        if lat and lon:
            lat_deg = self.convert_to_degrees(lat)
            lon_deg = self.convert_to_degrees(lon)
            if lat_ref == 'S':
                lat_deg = -lat_deg
            if lon_ref == 'W':
                lon_deg = -lon_deg
        else:
            lat_deg = lon_deg = None

        if alt is not None:
            # alt este un tuple (num, den)
            alt_value = alt[0] / alt[1]
            # dacă alt_ref == 1 înseamnă valoare negativă
            if alt_ref == 1:
                alt_value = -alt_value
        else:
            alt_value = None

        self.lat = lat_deg
        self.lon = lon_deg
        self.alt = alt_value


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} path_catre_imagine.jpg")
        sys.exit(1)

    path = sys.argv[1]
    drone_type = DroneType.DJI



