# imagery_math.py
import math
from typing import Tuple, List

METERS_PER_DEGREE = 111_320

def calculate_gsd(alt_m: float,
                  sensor_mm: float,
                  focal_mm: float,
                  img_px: int) -> float:
    """
    Compute ground sample distance (cm/pixel).
    """
    # m/pix → convert to cm/pix
    return (alt_m * sensor_mm) / (focal_mm * img_px) * 100

def latlon_to_pixel(lat: float, lon: float,
                    center_lat: float, center_lon: float,
                    direction_deg: float,
                    gsd_cm: float,
                    img_width: int, img_height: int
                   ) -> Tuple[float,float]:
    """
    Map a lat/lon to pixel coords relative to image center.
    """
    dy = (lat - center_lat) * METERS_PER_DEGREE
    dx = (lon - center_lon) * (METERS_PER_DEGREE * math.cos(math.radians(center_lat)))
    θ = math.radians(-direction_deg)
    x_m =  dx * math.cos(θ) - dy * math.sin(θ)
    y_m =  dx * math.sin(θ) + dy * math.cos(θ)
    return (img_width/2 + x_m/(gsd_cm/100),
            img_height/2 - y_m/(gsd_cm/100))

def image_corners(center_lat: float, center_lon: float,
                  direction_deg: float,
                  gsd_cm: float,
                  img_width: int, img_height: int,
                  adjust_for_mini4k: bool=False
                 ) -> List[Tuple[float,float]]:
    """
    Compute the four corner lat/lon coordinates of an image.
    """
    half_w = (gsd_cm * img_width / 100)/2
    half_h = (gsd_cm * img_height/100)/2
    diff = 90 - (20 if adjust_for_mini4k else 0)
    θ = math.radians(direction_deg - diff)

    raw = [(-half_w, half_h), (half_w, half_h),
           (half_w, -half_h), (-half_w, -half_h)]
    corners = []
    for x,y in raw:
        xr = x*math.cos(θ) - y*math.sin(θ)
        yr = x*math.sin(θ) + y*math.cos(θ)
        lat = center_lat + yr/METERS_PER_DEGREE
        lon = center_lon + xr/(METERS_PER_DEGREE*math.cos(math.radians(center_lat)))
        corners.append((lat, lon))
    return corners
