#!/bin/bash

# Activate licenta environment
source licenta/bin/activate

# Get the current date and time in yyyy_mm_dd_hh_mm format
timestamp=$(date +"%Y_%m_%d_%H_%M")
# Create a variable for the camera folder
cam_folder="captures_${timestamp}"

# Add the camera folder as an argument to the Python script
python3 capture_photo.py --cam_folder "${cam_folder}" > "${timestamp}.log"
mv "${timestamp}.log" "${cam_folder}/"
