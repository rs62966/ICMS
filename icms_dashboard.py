"""ICMS Dashboard Application

This application utilizes face recognition to monitor passengers in an aircraft cabin. It includes features such as seat mapping, face verification, and seatbelt status tracking.

Author: Ravi Shanker Singh
"""

import copy
import json
import pathlib
import sys
import tkinter as tk
from tkinter import PhotoImage

import cv2

from CameraAccess import create_webcam_stream
from database import get_passenger_data
from helper import (NotificationController, create_engine, do_face_verification, draw_seats,
                    process_faces, seats_coordinates, time_consumer)
from log import Logger
from joblib import Parallel, delayed


class Config:
    """Configuration class for ICMS Dashboard."""

    def __init__(self):
        """Initialize configuration parameters."""
        current = pathlib.Path(__file__).parent.resolve()
        self.background = current.joinpath("Images", "home.png")

        with open("config.json") as data_file:
            data = json.load(data_file)

        self.camera_source_1 = data["CAMERA"]["FIRST_CAMERA_INDEX"]
        self.camera_source_2 = data["CAMERA"]["SECOND_CAMERA_INDEX"]
        self.seat_coordinates = seats_coordinates(data["SEAT_COORDINATES"], data["FRAME_SHAPE"])


CONFIG = Config()
logger = Logger(module="ICMS Dashboard")
engine, r = create_engine()




# fmt: off
class WebcamApp:
    """Main class for the ICMS Dashboard application."""

    def __init__(self, root):
        """Initialize the application."""
        # Initialize the main application
        self.root = root
        self.root.title("Webcam Face Recognition")

        # Set up GUI elements
        self.bg_image = PhotoImage(file=CONFIG.background)
        self.root.geometry("1920x1200")
        self.bg_label = tk.Label(root, image=self.bg_image)
        self.bg_label.place(relwidth=1, relheight=1)

        # Initialize seat coordinates
        self.seat_coordinate = CONFIG.seat_coordinates

        # Load passenger data from the database
        load_database = get_passenger_data()
        self.database = {passenger["passenger_name"]: passenger["passenger_dataset"] for passenger in load_database}
        # Create NotificationController
        self.notification_controller = NotificationController(self.root, load_database)

        # Initialize variables
        self.vid = None
        self.frame = None
        self.monitoring = False
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.frame_process = 0
        self.first_process_frame = 5
        self.last_five_frames = {}
        self.track_last_five_frames = {}
        self.empty_skip_update_notification = 5
        self.ui_statbility = {"A1": 0, "A2": 0, "B1": 0, "B2": 0}

    @classmethod
    def speak(cls, audio):
        engine.say(audio)
        engine.runAndWait()

    def start_monitoring(self):
        """Start the monitoring process."""
        dataset = self.notification_controller.initialize_seat_info()
        logger.info(f"Database Loaded for {dataset}")
        message = f"Welcome to Cyient UAM"
        self.speak(message)
        if not self.monitoring:
            self.monitoring = True
        self.start_webcam()

    def start_webcam(self):
        """Start the webcam stream."""
        self.vid = create_webcam_stream(CONFIG.camera_source_1, CONFIG.camera_source_2)
        self.vid.start()
        self.show_frames()

    @time_consumer
    def show_frames(self):
        """Display frames from the webcam."""
        try:
            if self.vid.stopped:
                return

            self.frame = self.vid.read()
            if self.frame is not None:
                self.frame_process += 1

                # Process frames and store every seat face signature
                self.process_frames()

                # Display frames with seat ROI for seat mapping
                self.display_frames()

                # Log and track results every 'process_frame' frames
                if len(self.last_five_frames) == self.first_process_frame:
                    self.tracker()

                # Check for the 'q' key to stop the video stream
                key = cv2.waitKey(1)
                if key == ord("q"):
                    self.vid.stop()
                    cv2.destroyAllWindows()
                    self.notification_controller.initialize_seat_info()
                    return
            self.root.after(100, self.show_frames)
        except Exception as e:
            logger.error(f"Error in show_frames: {e}")

    def process_seat_info(self, face_embed):
        """Process seat information based on face embedding."""
        passenger_name, passenger_seat, match_distance = "", "", 0
        try:
            passenger_name, passenger_seat, match_distance = do_face_verification(self.database, face_embed)
        except Exception as e:
            logger.error(f"Error in process_seat_info: {e}")

        log_info = {
            "passenger_name": passenger_name,
            "passenger_assign_seat": passenger_seat,
            "passenger_match_distance": match_distance,
        }
        return log_info

    def update_gui(self):
        """Update the GUI based on seatbelt status."""
        empty_skip_notification = self.empty_skip_update_notification

        for seat, (name, status, color) in self.track_last_five_frames.items():
            if status == "Empty":
                self.ui_statbility[seat] += 1  # Initialize to 0 if it's None
                for key, value in self.ui_statbility.items():
                    if value == empty_skip_notification:
                        self.notification_controller.update_single_seat(seat, None, color, status)
                        self.ui_statbility[key] = 0
            else:
                message = f"Welcome to {seat} on {name} with Status of {status}"
                print(message)
                self.speak(message)
                self.notification_controller.update_single_seat(seat, None, color, status)
        self.clear_frames()

    def tracker(self):
        """Track passenger information over the last five frames."""
        try:
            analysis_result = self.notification_controller.analysis(self.last_five_frames)
            self.track_last_five_frames = copy.deepcopy(analysis_result)
            self.update_gui()
        
        except Exception as e:
            logger.error(f"Error in tracker: {e}")

    def clear_frames(self):
        """Clear tracked and last five frames."""
        self.last_five_frames.clear()
        self.track_last_five_frames.clear()
    
    def process_frames(self):
        """Process frames and store face signatures."""
        try:
            result = process_faces(self.frame, self.seat_coordinate)
            frame_info = {"A1": [], "A2": [], "B1": [], "B2": []}

            # Use joblib to parallelize the processing of faces
            parallel_result = Parallel(n_jobs=-1, prefer="threads")(
                delayed(self.process_seat_info)(passenger_face_embedding)
                for seat_name, passenger_face_embedding in result.items()
                if len(passenger_face_embedding) == 1
            )

            # Update frame_info with the parallel results
            for seat_name, log_info in zip(result.keys(), parallel_result):
                frame_info[seat_name].append(log_info)

            self.last_five_frames[self.frame_process] = frame_info
        except Exception as e:
            logger.error(f"Error in process_frames: {e}")

    def display_frames(self):
        """Display frames with seat status."""
        # cv2.namedWindow("Cabin monitoring", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("Cabin monitoring", 600, 300)
        draw_seats(self.frame, self.seat_coordinate)
        cv2.imshow("Cabin monitoring", self.frame)

    def on_closing(self):
        """Handle closing the application."""
        try:
            if self.vid:
                self.vid.stop()
            self.root.destroy()
        except Exception as e:
            logger.exception(e)
            self.root.destroy()
        else:
            sys.exit()


def main():
    """Main function to start the application."""
    root = tk.Tk()
    app = WebcamApp(root)

    def start_monitoring(event=None):
        app.start_monitoring()

    # Set up the "Start Monitoring" button
    start_button = tk.Button(
        root,
        text="Start Monitoring",
        command=start_monitoring,
        font=("Arial", 18, "bold"),
        bg="#04AA6D",
        fg="white",
    )
    start_button.place(relx=0.5, rely=0.9, anchor="center")

    # Bind space bar to the "Start Monitoring" button
    root.bind("<space>", start_monitoring)
    root.bind("<F1>", lambda event: root.attributes("-fullscreen", True))
    root.bind("<Escape>", lambda event: root.attributes("-fullscreen", False))

    root.mainloop()


if __name__ == "__main__":
    main()
