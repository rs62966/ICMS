import logging
import pathlib
import tkinter as tk
from collections import defaultdict
from copy import deepcopy
from io import BytesIO

import cv2
import numpy as np
from face_recognition import face_encodings, face_locations
from PIL import Image, ImageTk

current = pathlib.Path(__file__).parent.resolve()
face_img = current.joinpath("Images", "face_icon.png")
SEAT_BEAT = False

Empty =  0
CORRECT =  1
INCORRECT =  -1
UNAUTHORIZED = 101

class ColoredFormatter(logging.Formatter):
    COLORS = {
        "ERROR": ("bright_red", "bold"),
        "WARNING": ("yellow",),
        "INFO": ("blue",),
    }

    def format(self, record):
        log_string = super().format(record)
        if record.levelname in self.COLORS:
            log_string = colorstr(*self.COLORS[record.levelname], log_string)
        return log_string


def setup_logger(logger_name="ICMS"):
    # Check if the logger with the given name already exists
    existing_logger = logging.getLogger(logger_name)
    if existing_logger.handlers:
        # If handlers are already present, return the existing logger
        return existing_logger

    # If no handlers are present, set up a new logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(logger_name)
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter("%(levelname)s - %(message)s"))
    logger.addHandler(handler)

    return logger


# Set up logging
logger = setup_logger()

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
        self.rectangle_width = 140
        self.rectangle_height = 80
        self.rectangle_text = "Empty"

        # Temperatue lable disable
        # self.rectangle_canvas_temp = tk.Canvas(root, width=self.rectangle_width, height=self.rectangle_height, bg=self.rectangle_color)
        # self.rectangle_canvas_temp.place(relx=x_rel + 0.12, rely=y_rel, anchor="center")
        # self.body_temp_text = self.rectangle_canvas_temp.create_text(self.rectangle_width / 2, self.rectangle_height / 2 - 10, text="Body Temp", fill="black", font=("Arial", 10))
        # self.temp_text = self.rectangle_canvas_temp.create_text(self.rectangle_width / 2, self.rectangle_height / 2 + 10, text="98.4F", fill="black", font=("Arial", 14,'bold'))

        self.rectangle_canvas_status = tk.Canvas(root, width=self.rectangle_width, height=self.rectangle_height, bg=self.rectangle_color)
        self.rectangle_canvas_status.place(relx=x_rel + 0.07, rely=y_rel, anchor="center")
        self.status_text = self.rectangle_canvas_status.create_text(
            self.rectangle_width / 2,
            self.rectangle_height / 2,
            text=self.rectangle_text,
            fill="black",
            font=("Arial", 8, "bold"),
        )

    def change_rectangle_color(self, new_color, status):
        self.rectangle_canvas_status.config(bg=new_color)
        self.rectangle_text = status
        self.rectangle_canvas_status.itemconfig(self.status_text, text=status)


# fmt: off

class NotificationController:
    SEAT_NAMES = ["A1", "A2", "B1", "B2"]
    UNAUTHORIZED_NAMES = {"Unknown", "Un"}

    def __init__(self, root,dataset, frame_process=5):
        self.frame_process = frame_process
        self.dataset = dataset
        self.root = root
        self.seat_info = None
        self.seats = self.seat_layout()
        self.passenger_track = defaultdict(int)

    def seat_layout(self):
        seat_positions = [
            ("A1", face_img, 0.66, 0.2),
            ("A2", face_img, 0.26, 0.2),
            ("B1", face_img, 0.66, 0.58),
            ("B2", face_img, 0.26, 0.58),
        ]
        return {name: Seat(self.root, name, img, x, y) for name, img, x, y in seat_positions}

    def initialize_seat_info(self):
        """
        Initialize seat information based on the dataset.
        """
        default_seat_info = {
            "passenger_name": "",
            "status": "Empty",
            "color": "white",
            "profile_image": None,
            "passenger_embedding": None,
        }
        try:
                
            self.seat_info = {seat_name: deepcopy(default_seat_info) for seat_name in self.SEAT_NAMES}
            
            if self.dataset:
                for passenger in self.dataset:
                    passenger_name, passenger_seat, passenger_embedding = passenger["passenger_dataset"]
                    self.seat_info[passenger_seat]["profile_image"] = passenger["passenger_image"]
                    self.seat_info[passenger_seat]["passenger_name"] = passenger_name
                    self.seat_info[passenger_seat]["passenger_embedding"] = passenger_embedding
                    self.update_single_seat(self.seats[passenger_seat],image_data=passenger["passenger_image"])

        except Exception as e:
            logger.error("Database not able to fetch Data.",e)

        return {seat: passenger.get("passenger_name") for seat, passenger in self.seat_info.items()}

    def update_seat_info(self, frame_results):
        """
        Update seat information based on the frame results.
        """
        self.passenger_track.clear()
        name, status, score = None, "Empty", Empty
        for frame_no, four_seat_info in frame_results.items():
            for seat_name, passenger in four_seat_info.items():
                if passenger:  # Skip if there is no passenger data
                    passenger_info = passenger[0]
                    name, status = passenger_info.get("passenger_name", ""), "Unauthorized"

                    if name not in self.UNAUTHORIZED_NAMES:
                        if passenger_info["passenger_assign_seat"] == seat_name:
                            status = "Correct"
                            score = CORRECT
                        else:
                            status = "Incorrect" 
                            score = INCORRECT
                    else:
                        score = UNAUTHORIZED
                self.passenger_track[seat_name] =  (frame_no, name, status, score)

    def analysis(self, frame_results):
        """
        Analyze seat information and determine the status of passengers.
        """
        result = {}
        self.update_seat_info(frame_results)
        passenger_track = self.get_passenger_last_seen_frame()
      

        for seat, passenger in self.seat_info.items():
            passenger_name = passenger.get("passenger_name")
            status = "Empty"
            color = "white"

            if passenger_name:
                count = passenger_track.get(passenger_name, 0)

                if count >= 2:
                    status = "Correct"
                    color = "Yellow"
                elif count < 2 and count > 0:
                    status = "Incorrect"
                    color = "Orange"

            elif any(passenger_track.get(name, 0) > 0 for name in self.UNAUTHORIZED_NAMES):
                status = "Unauthorized"
                color = "Red"

            passenger["status"] = status
            passenger["color"] = color

            result[seat] = [passenger_name, status, color, count]

        return result

    def get_passenger_last_seen_frame(self):
        """
        Analyze persons with frames records for each passenger.
        """
        last_seen_frames = {}

        # Initialize variables to track the maximum count and corresponding status
        max_count = 0
        max_status = None

        for seat_name, seat_info in self.passenger_track.items():
            passenger_name, frame_no, status, count = seat_info

            # Check if the passenger has been seen more times than the current maximum
            if count > max_count:
                max_count = count
                max_status = status

            # Update the last seen frame for the passenger
            last_seen_frames[passenger_name] = {"frame_no": frame_no, "status": status, "count": count}

        # Return the result with the maximum count and corresponding status
        return last_seen_frames, {"max_count": max_count, "max_status": max_status}

    def update_single_seat(self, seat, image_data=None, rectangle_color="white", status="Empty"):
        """
        Update information for a seat wise
        """
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
    Draw seats on the given frame.
    """
    color = (0, 0, 255)  # Red color for rectangles and text

    for x1, y1, x2, y2, name in seat_coordinates:
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


def do_face_verification(database_faces_embed, passanger_face_embed, tolerance=0.6):
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
    logger.info(f"face_verification measure:: {passenger_info}")

    if min_distance > tolerance:
        return "Unknown", "Un", min_distance
    return passenger_info
