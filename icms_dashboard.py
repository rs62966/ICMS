import pathlib
from timeit import default_timer as timer

start = timer()


import copy
import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import tkinter as tk
from io import BytesIO
from tkinter import PhotoImage

import cv2
from PIL import Image, ImageTk

from CameraAccess import create_webcam_stream
from database import get_passenger_data
from helper import NotificationController, Seat, do_face_verification, draw_seats, process_faces, resize, setup_logger

# Set up logging
logger = setup_logger()

# Set up file paths and camera sources
current = pathlib.Path(__file__).parent.resolve()
background = current.joinpath("Images", "home.png")
with open(current.joinpath("config.json")) as data_file:
    data = json.load(data_file)
    camera_source_1 = data["CAMERA"]["FIRST_CAMERA_INDEX"]
    camera_source_2 = data["CAMERA"]["SECOND_CAMERA_INDEX"]
    seat_coordinates = [(*coords, seat_name) for seat_name, coords in data["SEAT_COORDINATES"].items()]
    logger.info(f"Loading FIRST_CAMERA_INDEX {camera_source_1} and SECOND_CAMERA_INDEX {camera_source_2}")

end = timer()
print("Elapsed time: " + str(end - start))


class WebcamApp:
    def __init__(self, root):
        # Initialize the main application
        self.root = root
        self.root.title("Webcam Face Recognition")

        # Set up GUI elements
        self.bg_image = PhotoImage(file=background)
        self.root.geometry("1920x1200")
        self.bg_label = tk.Label(root, image=self.bg_image)
        self.bg_label.place(relwidth=1, relheight=1)

        # Initialize seat coordinates
        self.seat_coordinate = seat_coordinates

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
        self.vid = create_webcam_stream(camera_source_1, camera_source_2)
        self.vid.start()
        self.show_frames()

    def show_frames(self, width=1800, height=900):
        try:
            if self.vid.stopped:
                return

            self.frame = self.vid.read()
            if self.frame is not None:
                self.frame_process += 1

                # Resize the frame to the specified width and height
                self.frame = resize(self.frame, width, height)

                # Process frames and store every seat face signature
                self.process_frames()

                # Display frames with seat ROI for seat mapping
                self.display_frames()

                # Log and track results every 'process_frame' frames
                if len(self.last_five_frames) == self.process_frame:
                    # Get the current seat information and analysis the frames
                    analysis = self.notification_controller.analysis(self.last_five_frames)
                    logger.info(f"Result of :: {self.frame_process} {analysis}")

                    # Track the last five frames and clear the buffer
                    self.track_last_five_frames = copy.deepcopy(self.last_five_frames)
                    self.last_five_frames.clear()
                    # Reset the frame process counter
                    self.frame_process = 0

                # Check for the 'q' key to stop the video stream
                key = cv2.waitKey(1)
                if key == ord("q"):
                    self.vid.stop()
                    cv2.destroyAllWindows()
                    return
            self.root.after(100, self.show_frames)
        except Exception as e:
            logger.error("Error in show_frames: %s", e)

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
            logger.error("Error in process_frames: %s", e)

    def process_seat_info(self, face_embed):
        passenger_name, passenger_seat, match_distance = "", "", 0
        try:
            passenger_name, passenger_seat, match_distance = do_face_verification(self.database, face_embed)
        except Exception as e:
            logger.error("Error in process_seat_info: %s", e)

        log_info = {"passenger_name": passenger_name, "passenger_assign_seat": passenger_seat, "passenger_match_distance": match_distance}
        return log_info

    def display_frames(self):
        # Display frames with seat status
        draw_seats(self.frame, self.seat_coordinate)
        cv2.imshow("Cabin monitoring", self.frame)

    def update_single_seat(self, seat, image_data=None, rectangle_color="white", status="Empty"):
        # Update information for a single seat
        if image_data:
            load_image = Image.open(BytesIO(image_data))
            tk_image = ImageTk.PhotoImage(load_image)
            seat.image_label.config(image=tk_image)
            seat.image_label.image = tk_image
        seat.change_rectangle_color(rectangle_color, status)

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
