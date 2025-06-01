#!/usr/bin/env python3

import os
import sys
import argparse
import ast

from PIL import Image, ImageDraw, ImageFont
from GPS_raw_entry import parse_capture_file, GPSRawInt
from elevation import process_coordinates

def get_corrected_gps_objects(folder_path: str):
    """
    1) Parse 'capture_gps_info.txt' into GPSRawInt objects.
    2) Extract coordinates and fetch ground elevations.
    3) Compute altitude_agl_m = altitude_m - ground_elevation for each object.
    Returns the list of GPSRawInt objects with new attribute 'altitude_agl_m'.
    """
    gps_objects = parse_capture_file(folder_path)
    if not gps_objects:
        print(f"[ERROR] No GPS entries found in '{os.path.join(folder_path, 'capture_gps_info.txt')}'.")
        return []

    # Build list of (lat, lon) in decimal degrees
    coordinates = [(obj.latitude_deg, obj.longitude_deg) for obj in gps_objects]

    try:
        ground_elevations = process_coordinates(coordinates)
    except Exception as e:
        print(f"[ERROR] process_coordinates failed: {e}")
        return []

    if len(ground_elevations) != len(gps_objects):
        print("[ERROR] Mismatch between GPS entries and returned elevations.")
        return []

    # Compute AGL altitude and attach as attribute
    for obj, gr in zip(gps_objects, ground_elevations):
        obj.altitude_agl_m = obj.altitude_m - gr

    return gps_objects

def parse_log_for_image_indices(log_path: str):
    """
    Reads the .log file at log_path and returns a dict mapping:
        { image_filename: gps_index (int) }.

    Algorithm (single-pass):
      - When a "[CAMERA] Captured .../<filename>" line appears, store filename as 'pending_image'.
      - When the next "[MAVLINK] Wrote ... - {idx}" line appears (and pending_image is not None),
        assign that idx to pending_image in the mapping, then clear pending_image.
    """
    mapping = {}
    pending_image = None

    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()

            if "[CAMERA] Captured" in line and ".jpg" in line:
                parts = line.split()
                for token in parts:
                    if token.lower().endswith(".jpg"):
                        pending_image = os.path.basename(token)
                        break
                continue

            if "[MAVLINK] Wrote" in line and '-' in line:
                parts = line.split('-')
                if len(parts) >= 2 and pending_image:
                    try:
                        idx = int(parts[-1].strip())
                        mapping[pending_image] = idx
                    except ValueError:
                        pass
                    pending_image = None
                continue

    return mapping

def annotate_images(folder_path: str, gps_objects: list):
    """
    Given the folder with images and the list of GPSRawInt objects (indexed by their capture order),
    parse the .log file to map each image to a GPS index, then open each JPEG,
    overlay 'lat, lon, alt_agl' onto the top-left corner, and save into 'annotated/'.
    """

    # Extract the date from the folder name (assuming format 'captures_{date}')
    folder_name = os.path.basename(folder_path)
    if folder_name.startswith("captures_"):
        date = folder_name.split("_", 1)[1]
        log_path = os.path.join(folder_path, f"{date}.log")
    else:
        print(f"[ERROR] Folder name '{folder_name}' does not follow the expected format 'captures_{date}'.")
        return
    if not os.path.isfile(log_path):
        print(f"[ERROR] Log file '{log_path}' not found.")
        return

    image_to_index = parse_log_for_image_indices(log_path)
    if not image_to_index:
        print(f"[WARNING] No image–index mappings found in log '{log_path}'.")
        return

    out_dir = os.path.join(folder_path, "annotated")
    os.makedirs(out_dir, exist_ok=True)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    for img_name, idx in image_to_index.items():
        img_path = os.path.join(folder_path, img_name)
        if not os.path.isfile(img_path):
            print(f"[WARNING] '{img_path}' not found; skipping.")
            continue

        if idx < 0 or idx >= len(gps_objects):
            print(f"[WARNING] Index {idx} for image '{img_name}' out of range; skipping.")
            continue

        gps_obj = gps_objects[idx]
        text = (
            f"lat: {gps_obj.latitude_deg:.6f}\n"
            f"lon: {gps_obj.longitude_deg:.6f}\n"
            f"alt AGL: {gps_obj.altitude_agl_m:.2f} m"
        )

        img = Image.open(img_path).convert("RGBA")
        draw = ImageDraw.Draw(img)

        x, y = 10, 10
        fill_color = (255, 255, 255, 255)
        outline_color = (0, 0, 0, 255)

        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            draw.text((x+dx, y+dy), text, font=font, fill=outline_color)

        draw.text((x, y), text, font=font, fill=fill_color)

        out_path = os.path.join(out_dir, img_name)
        img.convert("RGB").save(out_path, "JPEG")
        print(f"[OK] Annotated '{img_name}' with index {idx} → saved to '{out_path}'.")

def main():
    parser = argparse.ArgumentParser(
        description="Annotate JPEG captures with geolocation from GPS corrections."
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Path to the folder containing 'capture_gps_info.txt', 'run.log', and images."
    )
    args = parser.parse_args()
    folder_path = args.folder

    if not os.path.isdir(folder_path):
        print(f"[ERROR] '{folder_path}' is not a valid directory.")
        sys.exit(1)

    gps_objects = get_corrected_gps_objects(folder_path)
    if not gps_objects:
        sys.exit(1)

    annotate_images(folder_path, gps_objects)

if __name__ == "__main__":
    main()
