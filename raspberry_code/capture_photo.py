#!/usr/bin/env python3

import threading
import time
import os
import cv2
from picamera2 import Picamera2
from pymavlink import mavutil
import argparse

# Event for graceful shutdown of threads
stop_event = threading.Event()

def mavlink_thread(port, baud, output_file):
    # Initialize the MAVLink connection on the given serial port and baud rate
    master = mavutil.mavlink_connection(port, baud=baud)
    print(f"[MAVLINK] Connected to {port} at {baud} bps")

    # Wait for a heartbeat from the autopilot, up to 10 seconds
    hb = master.wait_heartbeat(timeout=10)
    if hb is None:
        print("[MAVLINK] No heartbeat in 10s - Stopping MAVLink thread")
        return
    print(f"[MAVLINK] Heartbeat received at {time.strftime('%Y-%m-%d %H:%M:%S')} - Starting GPS capture")
    
    # Define the MAVLink message types that carry GPS data
    GPS_MESSAGES = ['GPS_RAW_INT', 'HIL_GPS']
    # Open the output file in write mode
    with open(output_file, 'w') as f:
        index = 0
        # Loop until stop_event is set
        while not stop_event.is_set():
            # Block until a GPS message of the specified types arrives
            msg = master.recv_match(type=GPS_MESSAGES, blocking=True)
            if msg:
                # Convert the message to a dictionary for easy formatting
                d = msg.to_dict()
                # Format the line with an index, message type, and full dictionary
                line = f"[{index}] [{msg.get_type()}] {d}\n"
                f.write(line)
                # Print a simple log entry with a timestamp and index
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"[MAVLINK] Wrote {timestamp} - {index}")
                f.flush()
                index += 1

def camera_thread(output_folder, interval, count):
    """
    Configure Picamera2 and capture a fixed number of images at a set interval.
    """
    # Create Picamera2 instance 
    picam2 = Picamera2()
    # Ensure the output directory exists 
    os.makedirs(output_folder, exist_ok=True)

    # Configure the camera for still captures at 1280×720 resolution
    config = picam2.create_still_configuration(main={"size": (1280, 720)})
    picam2.configure(config)
    picam2.start()
    # Set fast exposure and a base analog gain for consistent image brightness
    picam2.set_controls({'ExposureTime': 5000, 'AnalogueGain': 1.0})
    print("[CAMERA] Started - capturing images...")

    for i in range(count):
        # Exit early if the stop_event was set from the main thread
        if stop_event.is_set():
            break

        # Capture image data from Picamera2 as a NumPy array
        image = picam2.capture_array("main")
        # Generate a timestamp string for naming the image file
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{output_folder}/capture_{ts}.jpg"
        # Write the image to disk in JPEG format
        cv2.imwrite(filename, image)
        print(f"[CAMERA] Captured {filename} at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Stop the camera when complete or when interrupted
    picam2.stop()
    print("[CAMERA] Stopped")

if __name__ == "__main__":
    # Parse command‐line arguments
    parser = argparse.ArgumentParser(description="Capture photos and GPS data.")
    parser.add_argument(
        "--cam_folder",
        type=str,
        required=True,
        help="Folder to save captured images."
    )
    args = parser.parse_args()

    # Define the serial port and baud rate for MAVLink 
    port = "/dev/ttyAMA0"
    baud = 115200
    # Set the camera output folder from command‐line argument
    cam_folder = args.cam_folder
    # Construct the GPS log filename inside the same folder
    gps_file = cam_folder + "/capture_gps_info.txt"

    # Determine how often to capture and how many images to take
    interval_s = 1        # seconds between captures
    n_images = 60         # total number of captures

    # Start the MAVLink and Camera threads
    t1 = threading.Thread(
        target=mavlink_thread,
        args=(port, baud, gps_file)
    )
    t2 = threading.Thread(
        target=camera_thread,
        args=(cam_folder, interval_s, n_images)
    )
    t1.start()
    t2.start()

    try:
        # Wait for the camera thread to finish 
        t2.join()
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted by user")
    finally:
        # Signal the MAVLink thread to stop and wait for it to exit
        stop_event.set()
        t1.join()
        print("[MAIN] All threads stopped. Exiting.")
