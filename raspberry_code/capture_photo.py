#!/usr/bin/env python3

import threading
import time
import os
import cv2
from picamera2 import Picamera2
from pymavlink import mavutil

# Event pentru oprirea ordonată a thread‑urilor
stop_event = threading.Event()

def mavlink_thread(port, baud, output_file):
    """
    Deschide conexiunea MAVLink și salvează doar mesajele GPS în output_file.
    """
    # inițializează legătura MAVLink
    master = mavutil.mavlink_connection(port, baud=baud)
    print(f"[MAVLINK] Connected to {port} at {baud} bps")
    master.wait_heartbeat()
    print(f"[MAVLINK] Heartbeat received at {time.strftime('%Y-%m-%d %H:%M:%S')} ▶ Starting GPS capture")
    
    # tipuri de mesaje GPS
    GPS_MESSAGES = ['GPS_RAW_INT', 'HIL_GPS']
    with open(output_file, 'w') as f:
        while not stop_event.is_set():
            msg = master.recv_match(type=GPS_MESSAGES, blocking=True)
            if msg:
                d = msg.to_dict()
                line = f"[{msg.get_type()}] {d}\n"
                #print(line, end='')
                f.write(line)
                f.flush()

def camera_thread(output_folder, interval, count):
    """
    Configurează Picamera2 și salvează count imagini cu intervalul dat.
    """
    picam2 = Picamera2()
    os.makedirs(output_folder, exist_ok=True)

    # configurare still capture la 1280×720
    config = picam2.create_still_configuration(main={"size": (1280, 720)})
    picam2.configure(config)
    picam2.start()
    # expunere rapidă și câștig analog
    picam2.set_controls({'ExposureTime': 5000, 'AnalogueGain': 1.0})
    print("[CAMERA] Started - capturing images...")

    for i in range(count):
        if stop_event.is_set():
            break
        image = picam2.capture_array("main")
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{output_folder}/capture_{ts}.jpg"
        cv2.imwrite(filename, image)
        print(f"[CAMERA] Captured {filename} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        time.sleep(interval)

    picam2.stop()
    print("[CAMERA] Stopped")

if __name__ == "__main__":
    # parametrizare
    port = "/dev/ttyAMA0"
    baud = 115200
    gps_file = "captureinfo.txt"
    cam_folder = "captures"
    interval_s = 1       # secunde între capturi
    n_images = 360       # număr de capturi

    # pornire thread‑uri
    t1 = threading.Thread(target=mavlink_thread, args=(port, baud, gps_file))
    t2 = threading.Thread(target=camera_thread, args=(cam_folder, interval_s, n_images))
    t1.start()
    t2.start()

    try:
        # așteaptă terminarea thread‑ului camerei
        t2.join()
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted by user")
    finally:
        # semnalează thread‑ului MAVLink să se oprească și așteaptă‑l
        stop_event.set()
        t1.join()
        print("[MAIN] All threads stopped. Exiting.")
