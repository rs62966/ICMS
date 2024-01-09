"""Helper of ICMS Application

This application utilizes face recognition to monitor passengers in an aircraft cabin. It includes features such as seat mapping, face verification, and seatbelt status tracking.

Author: Ravi Shanker Singh
"""

import logging
import os
import pathlib
import time
import tkinter as tk
from collections import Counter, defaultdict
from datetime import datetime
from io import BytesIO

import cv2
import numpy as np
from face_recognition import face_encodings, face_locations
from PIL import Image, ImageTk

from seatbelt import seatbelt_status

current = pathlib.Path(__file__).parent.resolve()
face_img = current.joinpath("Images", "face_icon.png")
SEAT_BEAT = False


class Logger:
    def __init__(self, module=None):
        self.module = module
        self.COLORS = {
            "INFO": ("blue",),
            "DEBUG": ("yellow",),
            "WARNING": ("bright_yellow",),
            "ERROR": ("bright_red", "bold"),
            "CRITICAL": ("red", "bold"),
        }
        log_level = os.environ.get("ICMS_LOG_LEVEL", str(logging.INFO))
        try:
            self.log_level = int(log_level)
        except Exception as err:
            self.dump_log(
                f"Exception while parsing $ICMS_LOG_LEVEL."
                f"Expected int but it is {log_level} ({str(err)})."
                "Setting app log level to info."
            )
            self.log_level = logging.INFO

    def info(self, message):
        if self.log_level <= logging.INFO:
            self.dump_log(f"{message}", level="INFO")

    def debug(self, message):
        if self.log_level <= logging.DEBUG:
            self.dump_log(f"🕷️ {message}", level="DEBUG")

    def warn(self, message):
        if self.log_level <= logging.WARNING:
            self.dump_log(f"⚠️ {message}", level="WARNING")

    def error(self, message):
        if self.log_level <= logging.ERROR:
            self.dump_log(f"🔴 {message}", level="ERROR")

    def critical(self, message):
        if self.log_level <= logging.CRITICAL:
            self.dump_log(f"💥 {message}", level="CRITICAL")

    def dump_log(self, message, level="INFO"):
        color_args = self.COLORS.get(level.upper(), ("blue",))
        colored_message = colorstr(*color_args, message)
        print(f"{str(datetime.now())[2:-7]} {self.module} - {colored_message}")


if SEAT_BEAT:
    from keras.models import load_model

    model_file = current.joinpath("model", "keras_model.h5")
    class_names = ["Seat Belt", "No Seat Belt"]
    model = load_model(model_file, compile=False)

    def seat_belt_process(frame):
        # Resize and preprocess the frame
        # image = resize(frame, 224, 224)
        image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        image = (image / 127.5) - 1
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(image)[0]

        # Get seat belt status and confidence score
        seat_belt_status = class_names[np.argmax(prediction)]
        confidence_score = prediction.max()

        return seat_belt_status, confidence_score


logger = Logger(module="Helper Module")


class Seat:
    def __init__(self, root, label, image_path, x_rel, y_rel):
        self.seat_name = label
        self.label = tk.Label(root, text=label, font=("Arial", 18), bg="#007D96", fg="white", width=5, height=3)
        self.label.place(relx=x_rel, rely=y_rel, anchor="center")
        self.image_label = tk.Label(root)
        self.image_label.place(relx=x_rel + 0.06, rely=y_rel + 0.17, anchor="center")
        image = Image.open(image_path)
        tk_image = ImageTk.PhotoImage(image)

        self.default_image = tk_image
        self.image_label.config(height=200, width=317, image=self.default_image)

        self.rectangle_color = "white"
        self.rectangle_width = 200
        self.rectangle_height = 80
        self.rectangle_text = "Empty"

        self.rectangle_canvas_status = tk.Canvas(root, width=self.rectangle_width, height=self.rectangle_height, bg=self.rectangle_color)
        self.rectangle_canvas_status.place(relx=x_rel + 0.09, rely=y_rel, anchor="center")
        self.status_text = self.rectangle_canvas_status.create_text(
            self.rectangle_width / 2,
            self.rectangle_height / 2,
            text=self.rectangle_text,
            fill="black",
            font=("Arial", 12, "bold"),
        )

    def change_rectangle_color(self, new_color, status):
        self.rectangle_canvas_status.config(bg=new_color)
        self.rectangle_text = status
        self.rectangle_canvas_status.itemconfig(self.status_text, text=status)


def seats_coordinates(data, frame_shape):
    h, w, d = frame_shape
    return [(int(coord[0] * w), int(coord[1] * h), int(coord[2] * w), int(coord[3] * h), seat_name) for seat_name, coord in data.items()]


# fmt: off
class NotificationController:
    """
    A class for managing notifications related to seat assignments and belt statuses.

    Attributes:
        ALLOWED_SEAT_NAMES (list): List of allowed seat names.

    Methods:
        __init__(self, root, dataset): Initialize the NotificationController.
        belt_read(self): Read the status of seat belts.
        initialize_seats(self): Initialize seat objects.
        initialize_seat_info(self): Initialize seat information.
        update_seat_info(self, frame_results): Update seat information based on frame results.
        analyze_frames(self, data): Analyze frame results.
        analysis(self, frame_results): Perform analysis on frame results.
        update_single_seat(self, update_seat, image_data=None, rectangle_color="white", status="Empty"): Update a single seat.

    """
    ALLOWED_SEAT_NAMES = ["A1", "A2", "B1", "B2"]
    UNAUTHORIZED_NAMES = {"Unknown", "Un"}

    def __init__(self, root, dataset):
        """
        Initialize the NotificationController.

        Args:
            root (object): The root object for the GUI.
            dataset (list): List of passenger data.

        """
        self.dataset = dataset
        self.root = root
        self.seats = self.initialize_seats()
        self.seat_info = None
        self.passenger_track = defaultdict(list)

    def belt_read(self):
        """
        Read the status of seat belts.

        Returns:
            dict: Dictionary with seat names as keys and seat belt status as values.

        """
        try:
            seat_belt_status = seatbelt_status()
        except Exception as e:
            seat_belt_status = {seat: False for seat in self.ALLOWED_SEAT_NAMES}
        return seat_belt_status

    def initialize_seats(self):
        """
        Initialize seat objects.

        Returns:
            dict: Dictionary with seat names as keys and Seat objects as values.

        """
        seat_positions = [
            ("A1", face_img, 0.66, 0.2),
            ("A2", face_img, 0.26, 0.2),
            ("B1", face_img, 0.66, 0.58),
            ("B2", face_img, 0.26, 0.58),
        ]
        return {name: Seat(self.root, name, img, x, y) for name, img, x, y in seat_positions}

    def initialize_seat_info(self):
        """
        Initialize seat information.

        Returns:
            dict: Dictionary with seat names as keys and passenger names as values.

        """
        default_seat_info = {
            "passenger_name": "",
            "status": "Empty",
            "color": "white",
            "profile_image": None,
            "passenger_embedding": None,
        }

        self.seat_info = {seat_name: default_seat_info.copy() for seat_name in self.ALLOWED_SEAT_NAMES}
        if self.dataset:
            for passenger in self.dataset:
                passenger_name, passenger_seat, passenger_embedding = passenger["passenger_dataset"]
                seat = self.seat_info[passenger_seat]
                seat.update({
                    "profile_image": passenger["passenger_image"],
                    "passenger_name": passenger_name,
                    "passenger_embedding": passenger_embedding,
                })
                self.update_single_seat(self.seats[passenger_seat], image_data=passenger["passenger_image"])
        else:
            logger.error("Database not able to fetch Data.")

        return {seat: passenger.get("passenger_name") for seat, passenger in self.seat_info.items()}

    def update_seat_info(self, frame_results):
        """
        Update seat information based on frame results.

        Args:
            frame_results (dict): Dictionary with seat names as keys and frame results as values.

        """
        for _, seat_info in frame_results.items():
            belt_data = self.belt_read()
            for seat_name, passengers in seat_info.items():
                if passengers:
                    passenger_info = passengers[0]
                    name = passenger_info.get("passenger_name", "")
                    status = "Correct" if name not in self.UNAUTHORIZED_NAMES and passenger_info["passenger_assign_seat"] == seat_name else "Incorrect" if name not in self.UNAUTHORIZED_NAMES else "Unauthorized"
                    belt_status = belt_data.get(seat_name, False)
                    color = 'green' if belt_status and status == 'Correct' else 'yellow' if belt_status else 'orange' if status == 'Incorrect' else 'red'
                else:
                    name, status, color = "", "Empty", "white"
                    belt_status = belt_data.get(seat_name, False)
                    color = 'yellow' if belt_status else 'white'

            self.passenger_track[seat_name].append((name, status, color))
    def analyze_frames(self):
        """
        Analyze frame results.

        Args:
            data (dict): Dictionary with seat names as keys and values as lists of frame results.

        Returns:
            dict: Dictionary with seat names as keys and analyzed results as values.

        """
    
        result = {seat: max(Counter(values).items(), key=lambda x: x[1])[0] for seat, values in self.passenger_track.items()}
        return result

    def analysis(self, frame_results):
        """
        Perform analysis on frame results.

        Args:
            frame_results (dict): Dictionary with seat names as keys and frame results as values.

        Returns:
            dict: Dictionary with seat names as keys and analyzed results as values.

        """
        self.passenger_track.clear()
        self.update_seat_info(frame_results)
        result = self.analyze_frames()
        # belt_data = self.belt_read()
        
        # updated_results = {
        #     seat: tuple(seat_info[:-2]) + ("Ready","green") if seat_info[-1] == "yellow" and belt_data[seat] else seat_info
        #     for seat, seat_info in result.items()
        # }
        
        return result


    def update_single_seat(self, update_seat, image_data=None, rectangle_color="white", status="Empty"):
        """
        Update a single seat.

        Args:
            update_seat (str or object): Seat name or Seat object to be updated.
            image_data (bytes, optional): Image data for updating the seat's image.
            rectangle_color (str, optional): Color of the rectangle.
            status (str, optional): Status of the seat.

        """
        seat = self.seats[update_seat] if isinstance(update_seat, str) else update_seat

        if image_data:
            load_image = Image.open(BytesIO(image_data))
            tk_image = ImageTk.PhotoImage(load_image)
            seat.image_label.config(image=tk_image)
            seat.image_label.image = tk_image

        seat.change_rectangle_color(rectangle_color, status)



def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


def process_faces(frame, seat_coordinates):
    """
    Process faces in the given frame and return a dictionary with seat information.
    """
    processed_seats = {}

    for x1, y1, x2, y2, seat_name in seat_coordinates:
        seat_roi = frame[y2:y1, x1:x2]
        rgb_seat_roi = cv2.cvtColor(seat_roi, cv2.COLOR_BGR2RGB)
        face_area = face_locations(rgb_seat_roi)

        if len(face_area) == 1:
            face_encoding = face_encodings(rgb_seat_roi, face_area)
            processed_seats[seat_name] = face_encoding
        else:
            processed_seats[seat_name] = []

    return processed_seats


def draw_seats(frame, seat_coordinates):
    """
    Draw seats on the given frame with different colors.
    """
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Red, Green, Blue, Yellow

    for (x1, y1, x2, y2, name), color in zip(seat_coordinates, colors):
        cv2.rectangle(frame, (x1, y2), (x2, y1), color, 5)
        cv2.putText(frame, f"Seat {name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    return frame


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Resize the given image to the specified width and height.
    """
    if width is None and height is None:
        return image

    r = width / float(image.shape[1]) if width is not None else height / float(image.shape[0])
    dim = (width, int(image.shape[0] * r)) if width is not None else (int(image.shape[1] * r), height)
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def do_face_verification(database_faces_embed, passanger_face_embed, tolerance=0.55):
    """
    Perform face verification by comparing the embedding vectors from the database.
    """
    distances = np.zeros(len(database_faces_embed))
    data_point = {}
    for i, (passenger_name, passenger_data) in enumerate(database_faces_embed.items()):
        passenger_name, passenger_seat, passenger_embedding = passenger_data
        distances[i] = np.linalg.norm(np.array(passanger_face_embed) - np.array(passenger_embedding))
        data_point[i] = (passenger_name, passenger_seat, distances[i])

    idx = np.argmin(distances)
    min_distance = distances[idx]
    passenger_info = data_point[idx]
    logger.debug(f"face_verification measure:: {passenger_info}")

    if min_distance > tolerance:
        return "Unknown", "Un", min_distance
    return passenger_info


def time_consumer(func):
    """_summary_

    Args:
        func (_type_): _description_

    Returns:
        _type_: _description_
    """
    # This function shows the execution time of

    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f"Function {func.__name__!r} executed in {(t2-t1):.4f}s")
        return result

    return wrap_func
