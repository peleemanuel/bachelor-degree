#!/usr/bin/env python3
import sys
import piexif
from enum import Enum

class DroneType(Enum):
    SELF_MADE_DRONE = "selfMadeDrone"
    DJI = "DJI"
    DJI_MINI_4K = "DJIMini4K"

class DroneInteface():
    def __init__(self, drone_type: DroneType, sensor_width: float,focal_length: float):
        self.drone_type = drone_type # Drone Type
        self.sensor_width = sensor_width # Sensor width in mm
        self.focal_length = focal_length # Focal length in mm

    def __str__(self):
        return f"DroneInteface(drone_type={self.drone_type}, sensor_width={self.sensor_width}, focal_length={self.focal_length})"

class SelfMadeDrone(DroneInteface):
    def __init__(self):
        super().__init__(DroneType.SELF_MADE_DRONE, 5.02 , 6)

class DJI(DroneInteface):
    def __init__(self, sensor_width: float, focal_length: float):
        super().__init__(DroneType.DJI, sensor_width, focal_length)

class DJIMini4K(DroneInteface):
    def __init__(self):
        super().__init__(DroneType.DJI_MINI_4K, 6.3 / 4, 4.5)

class GPSInfo():
    """
    Class that will save the GPS information from an image and its path
    """
    def __init__(self, path: str, drone_type: DroneType, secondary_file: str = None):
        self.path = path
        self.lat = None # Latitude in degrees
        self.lon = None # Longitude in degrees
        self.alt = None # Altitude in meters
        self.image_direction = 0 # Image direction in degrees
        self.secondary_file = secondary_file # Secondary file path for self-made drone
        self.drone_type = drone_type # Drone type

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
        elif self.drone_type == DroneType.DJI or self.drone_type == DroneType.DJI_MINI_4K:
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

        img_direction = gps_ifd.get(piexif.GPSIFD.GPSImgDirection, None)

        if img_direction:
            self.image_direction = img_direction[0] / img_direction[1]

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
            alt_value = alt[0] / alt[1]
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

    gps_info = GPSInfo(path, drone_type)
    gps_info.extract_gps_info()
    print(gps_info)


