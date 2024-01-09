import subprocess
import os
import platform
import sys
import json
import pathlib
import cv2
from database import get_passenger_data
from helper import Logger
from seatbelt import seatbelt_status

logger = Logger(module="Check ICMS Dependency")

def print_separator():
    logger.info("*" * 30)

def check_camera():
    try:
        current = pathlib.Path(__file__).parent.resolve()
        with open(current.joinpath("config.json")) as data_file:
            data = json.load(data_file)

        camera_source_1 = data["CAMERA"]["FIRST_CAMERA_INDEX"]
        camera_source_2 = data["CAMERA"]["SECOND_CAMERA_INDEX"]

        for camera_source in [camera_source_1, camera_source_2]:
            cap = cv2.VideoCapture(camera_source)
            if not cap.isOpened():
                logger.info(f"Error: Camera source {camera_source} is not available.")
            else:
                logger.success(f"Camera source {camera_source} is available.")
                cap.release()

    except Exception as e:
        logger.error(f"Error checking camera devices: {e}")

def check_db():
    connection = None
    logger.info("🗄️  DB Status:")
    try:
        database =  get_passenger_data()
        logger.success(f"Successfully connected to the database and read passenger {len(database)}.")
    except Exception as e:
        logger.error(f"Error connecting to the database: {e}")
    finally:
        if connection is not None and connection.is_connected():
            connection.close()

def check_platform():
    logger.info("💻🖱️ Platform Info:")
    current_platform = platform.system()
    if current_platform.lower() == 'linux' and os.uname().machine.startswith('aarch64'):
        logger.success("It's a Jetson platform: {}".format(current_platform))
    else:
        logger.error(f"It's a {current_platform} platform.")

def check_installation(package):
    try:
        subprocess.check_output([sys.executable, "-m", "pip", "show", package])
        logger.success(f"✅ {package} is already installed.")
    except subprocess.CalledProcessError:
        subprocess.call([sys.executable, "-m", "pip", "install", package])
        logger.info(f"📦 {package} has been installed.")

def check_belt_read():
    try:
        seat_belt_status = seatbelt_status()
        logger.success("✅ Seat belt Sensor Working fine")
    except Exception as e:
        logger.error("❌ Seat belt Sensor Not Working")
    return seat_belt_status

def main():
    with open("requirements.txt", "r") as requirements_file:
        packages = requirements_file.read().splitlines()

    logger.info("🚀 Checking and installing required packages...")
    for package in packages:
        check_installation(package)

    print_separator()
    check_camera()
    print_separator()
    check_platform()
    print_separator()
    check_belt_read()
    print_separator()
    check_db()

if __name__ == "__main__":
    main()
