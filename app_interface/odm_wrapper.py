import subprocess
import sys
import os

def run_odm(image_folder):
    odm_exe = r"E:\licenta\bachelor-degree\app_interface\CloudODM\odm.exe"
    if not os.path.isdir(image_folder):
        print(f"Error: '{image_folder}' is not a valid directory.")
        return
    try:
        subprocess.run([odm_exe, image_folder], check=True)
    except subprocess.CalledProcessError as e:
        print(f"ODM execution failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python odm_wrapper.py <path_to_image_folder>")
    else:
        run_odm(sys.argv[1])