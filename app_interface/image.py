#!/usr/bin/env python3
from gps_info import GPSInfo, DroneType, SelfMadeDrone, DJI, DJIMini4K, DroneInteface
from PIL import Image
import sys
import math
import folium

METERS_PER_DEGREE = 111320  

def get_drone(drone_type: DroneType) -> DroneInteface:
    """
    Function that will return the drone object based on the drone type
    :param drone_type: The type of the drone
    :return: The drone object
    """
    if drone_type == DroneType.SELF_MADE_DRONE:
        return SelfMadeDrone()
    elif drone_type == DroneType.DJI:
        return DJI(6.3, 4.5)  # Example values for sensor width and focal length
    elif drone_type == DroneType.DJI_MINI_4K:
        return DJIMini4K()
    else:
        raise ValueError("Unknown drone type")

def GSD_calculator(drone: DroneInteface, gps_info: GPSInfo):
    """
    Function that will calculate the GSD of an image
    :return: GSD in cm/pixel
    """
    # Take the photo path from the GPSInfo object
    photo_path = gps_info.path
    # Get the image size
    img = Image.open(photo_path)
    width, _ = img.size

    # Caclulate the GSD
    # altitude in meters, sensor width in mm, focal length in mm, image width in pixels, final GSD in m/pixel
    GSD = (gps_info.alt * drone.sensor_width) / (drone.focal_length * width)

    # Convert to cm/pixel
    GSD = GSD * 100
    return GSD

def latlon_to_pixel(lat, lon, gps_info, gsd, img_width, img_height):
    """
    Convert a lat/lon point (or any X/Y in same units) into pixel (x, y)
    relative to the center of img of size (img_width, img_height).
    - gsd: ground‐sample‐distance in cm/pixel
    - gps_info.lat/lon: center of image
    - gps_info.image_direction: rotation in degrees from north
    """
    # 1) compute meter offsets from center
    dy_m = (lat - gps_info.lat) * 111320
    dx_m = (lon - gps_info.lon) * (111320 * math.cos(math.radians(gps_info.lat)))
    # 2) rotate into image axes
    θ = math.radians(-gps_info.image_direction)
    x_m =  dx_m * math.cos(θ) - dy_m * math.sin(θ)
    y_m =  dx_m * math.sin(θ) + dy_m * math.cos(θ)
    # 3) convert meters → pixels (gsd cm/pixel → m/pixel = gsd/100)
    px = img_width / 2  + (x_m / (gsd / 100))
    py = img_height / 2 - (y_m / (gsd / 100))
    return px, py

def get_image_corner_coordinates(GSD: float, gps_info: GPSInfo):
    """
    Function that will calculate the image corner coordinates
    :param GSD: GSD in cm/pixel
    :param gps_info: GPSInfo object
    :return: The image corner coordinates
    """
    # Take the photo path from the GPSInfo object
    photo_path = gps_info.path
    # Get the image size
    img = Image.open(photo_path)
    width, height = img.size
    ground_width = (GSD * width / 100) / 2
    ground_height = (GSD * height / 100) / 2

    difference = 90
    # If DJI Mini 4K, the image direction needs to be adjusted
    if gps_info.drone_type == DroneType.DJI_MINI_4K:
        difference -= 20

    tetha = math.radians(gps_info.image_direction - difference)

    corners = [
        (-ground_width, ground_height),  # Top-left
        (ground_width, ground_height),   # Top-right
        (ground_width, -ground_height),  # Bottom-right
        (-ground_width, -ground_height)  # Bottom-left
    ]

    corner_coords = []
    for x, y in corners:
        x_rotated = x * math.cos(tetha) - y * math.sin(tetha)
        y_rotated = x * math.sin(tetha) + y * math.cos(tetha)

        # Degree conversion
        delta_lat = y_rotated / METERS_PER_DEGREE
        delta_long = x_rotated / (METERS_PER_DEGREE * math.cos(math.radians(gps_info.lat)))

        lat = gps_info.lat + delta_lat
        lon = gps_info.lon + delta_long
        corner_coords.append((lat, lon))
    
    return corner_coords

def main():
    # Example usage
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} image_path.jpg")
        sys.exit(1)

    path = sys.argv[1]
    drone_type = DroneType.DJI_MINI_4K

    gps_info = GPSInfo(path, drone_type)
    gps_info.extract_gps_info()
    drone = get_drone(drone_type)

    gsd = GSD_calculator(drone, gps_info)
    print(f"GSD: {gsd:.2f} cm/pixel")
    corner_coords = get_image_corner_coordinates(gsd, gps_info)
    print("Image corner coordinates:")
    for i, (lat, lon) in enumerate(corner_coords):
        print(f"Corner {i + 1}: Latitude: {lat:.6f}, Longitude: {lon:.6f}")

    m = folium.Map(location=[gps_info.lat, gps_info.lon], zoom_start=17)

    folium.Polygon(
        locations=corner_coords,
        color='blue',
        fill=True,
        fill_opacity=0.5
    ).add_to(m)

    # Save map
    m.save("map.html")

if __name__ == '__main__':
    main()