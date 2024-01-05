import copy
import json
import pathlib
import sys
import tkinter as tk
from tkinter import PhotoImage

import cv2

from CameraAccess import create_webcam_stream
from database import get_passenger_data
from helper import (Logger, NotificationController, do_face_verification,
                    draw_seats, process_faces, seats_coordinates,
                    time_consumer)
from seatbelt import seatbelt_status

# Configuration


class Config:
    def __init__(self):
        current = pathlib.Path(__file__).parent.resolve()
        self.background = current.joinpath("Images", "home.png")

        with open(current.joinpath("config.json")) as data_file:
            data = json.load(data_file)

        self.camera_source_1 = data["CAMERA"]["FIRST_CAMERA_INDEX"]
        self.camera_source_2 = data["CAMERA"]["SECOND_CAMERA_INDEX"]
        self.seat_coordinates = seats_coordinates(data["SEAT_COORDINATES"], data["FRAME_SHAPE"])


CONFIG = Config()
logger = Logger(module="ICMS Dashboard")


class WebcamApp:
    def __init__(self, root):
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
        # Create of NotificationController
        self.notification_controller = NotificationController(self.root, load_database)

        # Initialize variables
        self.vid = None
        self.frame = None
        self.monitoring = False
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.frame_process = 0
        self.process_frame = 5
        self.last_five_frames = {}
        self.track_last_five_frames = {}

    def start_monitoring(self):
        # Start the monitoring process and Update the seat
        dataset = self.notification_controller.initialize_seat_info()
        logger.info(f"Database Loaded for {dataset}")
        if not self.monitoring:
            self.monitoring = True
        self.start_webcam()

    def start_webcam(self):
        # Start the webcam stream
        self.vid = create_webcam_stream(CONFIG.camera_source_1, CONFIG.camera_source_2)
        self.vid.start()
        self.show_frames()

    # @time_consumer
    def show_frames(self):
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
                if len(self.last_five_frames) == self.process_frame:
                    self.tracker()

                # Check for the 'q' key to stop the video stream
                key = cv2.waitKey(1)
                if key == ord("q"):
                    self.vid.stop()
                    cv2.destroyAllWindows()
                    return
            self.root.after(100, self.show_frames)
        except Exception as e:
            logger.error(f"Error in show_frames: {e}")

    def process_seat_info(self, face_embed):
        passenger_name, passenger_seat, match_distance = "", "", 0
        try:
            passenger_name, passenger_seat, match_distance = do_face_verification(self.database, face_embed)
        except Exception as e:
            logger.error(f"Error in process_seat_info: {e}")

        log_info = {"passenger_name": passenger_name, "passenger_assign_seat": passenger_seat, "passenger_match_distance": match_distance}
        return log_info

    def update_gui(self):
        try:
            seat_belt_status = seatbelt_status()
        except Exception as e:
            seat_belt_status = {"A1": False, "A2": False, "B1": False, "B2": False}

        for seat, passenger_info in self.track_last_five_frames.items():
            status, status_color = (("Ready", "Green") if passenger_info.get("color") == "Yellow" and seat_belt_status.get(seat, False) else (passenger_info.get("status"), passenger_info.get("color")))
            self.notification_controller.update_single_seat(seat, None, status_color, status)
        # Reset the tracker buffer
        self.track_last_five_frames.clear()

    def tracker(self):
        try:
            # Get analysis results
            analysis_result = self.notification_controller.analysis(self.last_five_frames)

            # Iterate over seat and result pairs
            for seat, result in analysis_result.items():
                # Check if the seat is present in the tracking history
                if seat in self.track_last_five_frames:
                    # Check if the passenger_name is the same
                    if self.track_last_five_frames[seat]['passenger_name'] == result['passenger_name']:
                        # Increment seen value if the passenger_name is the same
                        self.track_last_five_frames[seat]['seen'] = self.track_last_five_frames[seat].get('seen', 0) + 1
                    else:
                        # Update passenger_name and set seen to 1 for different passenger_name
                        self.track_last_five_frames[seat]['passenger_name'] = result['passenger_name']
                        self.track_last_five_frames[seat]['seen'] = 1
                else:
                    # If the seat is not present in track_last_five_frames, add it
                    self.track_last_five_frames[seat] = {
                        'color': result['color'],
                        'passenger_name': result['passenger_name'],
                        'score_count': 0,
                        'status': result['status'],
                        'seen': 1
                    }
            # Check if all 'seen' values are equal to 3
            all_seen_3 = all(item['seen'] == 3 for item in self.track_last_five_frames.values())
            logger.error(f"all_seen_3 {all_seen_3} ---->{self.track_last_five_frames.values()}")
            if all_seen_3:
                # Update the GUI
                self.update_gui()
                self.last_five_frames.clear()

            # Clear the last five frames
            self.last_five_frames.clear()

        except Exception as e:
            logger.error(f"Error in tracker: {e}")


    def process_frames(self):
        # Process the frames and store the information
        try:
            result = process_faces(self.frame, self.seat_coordinate)
            frame_info = {"A1": [], "A2": [], "B1": [], "B2": []}

            for seat_name, passenger_face_embedding in result.items():
                if len(passenger_face_embedding) == 1:
                    log_info = self.process_seat_info(passenger_face_embedding)
                    frame_info[seat_name].append(log_info)
            self.last_five_frames[self.frame_process] = frame_info
        except Exception as e:
            logger.error(f"Error in process_frames: {e}")

    def display_frames(self):
        # Display frames with seat status
        cv2.namedWindow("Cabin monitoring", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Cabin monitoring", 600, 300)  #
        draw_seats(self.frame, self.seat_coordinate)
        cv2.imshow("Cabin monitoring", self.frame)

    def on_closing(self):
        """
        Handle closing the application.
        """
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
    # Main function to start the application
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
