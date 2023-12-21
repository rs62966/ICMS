import os

# Set environment variable to suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import base64
import logging
import pathlib
import platform
import sys
import tkinter as tk
from io import BytesIO
from threading import Thread
from tkinter import PhotoImage, messagebox

import cv2
from face_recognition import face_locations, face_encodings
import mysql.connector
import numpy as np
from dotenv import load_dotenv
from keras.models import load_model
from PIL import Image, ImageTk
import time

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure NumPy options
np.set_printoptions(suppress=True)

# Define file paths
current = pathlib.Path(__file__).parent.resolve()
EMBED_N = 128
face_img = current.joinpath("Images", "face_icon.png")
background = current.joinpath("Images", "home.png")
camera_source_1 = 0
camera_source_2 = 1
font = cv2.FONT_HERSHEY_SIMPLEX
lineType = cv2.LINE_AA

# Access environment variables for database configuration
db_config = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_DATABASE"),
    "use_pure": os.getenv(
        "DB_USE_PURE", True
    ),  # Use pure Python implementation, default to True if not provided
}

# Load the Keras model
model = load_model("keras_model.h5", compile=False)

# Define class names
class_names = ["Seat Belt", "No Seat Belt"]


class WebcamStream:
    def __init__(self, stream_id=0):
        self.stream_id = stream_id
        if platform.system() == "Windows":
            self.vcap = cv2.VideoCapture(stream_id, cv2.CAP_DSHOW)
        else:
            self.vcap = cv2.VideoCapture(stream_id)

        self.grabbed, self.frame = self.vcap.read()

        if self.grabbed is False and self.vcap.isOpened() is False:
            print(f"[Exiting] No more frames to read camera index {stream_id} ")
            exit(0)

        self.stopped = True
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True

    def start(self):
        self.stopped = False
        self.t.start()

    def update(self):
        while True:
            if self.stopped is True:
                break
            self.grabbed, self.frame = self.vcap.read()
            if self.grabbed is False:
                print("[Exiting] No more frames to read")
                self.stopped = True
                break
        self.vcap.release()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


class DualWebcamStream:
    def __init__(self, stream_id1=0, stream_id2=1):
        self.stopped = True
        self.stream1 = WebcamStream(stream_id1)
        self.stream2 = WebcamStream(stream_id2)

    def start(self):
        self.stopped = False
        self.stream1.start()
        self.stream2.start()

    def read(self):
        frame1 = self.stream1.read()
        frame2 = self.stream2.read()

        if frame1 is not None and frame2 is not None:
            combined_frame = cv2.hconcat([frame1, frame2])
            return combined_frame
        else:
            return None

    def stop(self):
        self.stream1.stop()
        self.stream2.stop()
        self.stopped = True


def create_webcam_stream(*args):
    num_cameras = len(args)
    if num_cameras == 1:
        return WebcamStream(args[0])
    elif num_cameras == 2:
        return DualWebcamStream(args[0], args[1])
    else:
        raise ValueError("You can provide one or two camera IDs only.")


def process_faces(frame, seat_coordinates):
    """
    Process faces in the given frame and return a list of seat information.
    """
    processed_seats = []

    for x1, y1, x2, y2, name in seat_coordinates:
        seat_roi = frame[y2:y1, x1:x2]
        rgb_seat_roi = cv2.cvtColor(seat_roi, cv2.COLOR_BGR2RGB)

        face_location = face_locations(rgb_seat_roi)
        if len(face_location) == 1:
            face_encoding = face_encodings(rgb_seat_roi, face_location)
            processed_seats.append((name, face_encoding))
        else:
            processed_seats.append((name, []))  # No face or multiple faces detected

    return processed_seats


def draw_seats(frame, seat_coordinates):
    """
    Draw seats on the given frame.
    """
    color = (0, 0, 255)  # Red color for rectangles and text

    for x1, y1, x2, y2, name in seat_coordinates:
        cv2.rectangle(frame, (x1, y2), (x2, y1), color, 5)
        cv2.putText(frame,f"Seat {name}",(x1, y1 - 10),font,0.5,color,2,lineType,)

    return frame


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Resize the given image to the specified width and height.
    """
    if width is None and height is None:
        return image

    r = ( width / float(image.shape[1]) if width is not None else height / float(image.shape[0]))
    dim = ( (width, int(image.shape[0] * r)) if width is not None else (int(image.shape[1] * r), height) )
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def do_face_verification(embed_dict, embed, tolerance=0.6):
    """
    Perform face verification by comparing the embedding vectors from the database.
    """
    distances = np.zeros(len(embed_dict))
    data_point = {}
    for i, (passenger_name, passenger_data) in enumerate(embed_dict.items()):
        passenger_name, passenger_seat, passenger_embedding = passenger_data
        distances[i] = np.linalg.norm(np.array(embed) - np.array(passenger_embedding))
        data_point[i] = (passenger_name, passenger_seat, distances[i])

    idx = np.argmin(distances)
    min_distance = distances[idx]
    passenger_info = data_point[idx]
    logger.info(f"face_verification measure:: {passenger_info}")

    if min_distance > tolerance:
        return "Unknown", "Un", min_distance
    return passenger_info


def get_passenger_data():
    data_from_db = []

    connection = mysql.connector.connect(**db_config)
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT passengerName, personImage, passengerSeat FROM passengerdetails")
        results = cursor.fetchall()

        for passenger in results:
            try:
                load_image = base64.b64decode(passenger[1])
                image_np = np.frombuffer(load_image, dtype=np.uint8)
                image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
                face_encoding = face_encodings(image)[0]

                passenger_data = {
                    "passenger_name": passenger[0],
                    "passenger_assign_seat": passenger[2],
                    "passenger_image": load_image,
                    "passenger_encoding": face_encoding,
                }

                data_from_db.append(passenger_data)

            except Exception as e:
                logger.error("Error processing passenger %s: %s", passenger[0], e)
    except Exception as e:
        logger.error("Error fetching data from the database: %s", e)
    else:
        connection.close()

    return data_from_db


class Seat:
    def __init__(self, root, label, image_path, x_rel, y_rel):
        self.seat_name = label
        self.label = tk.Label(
            root,
            text=label,
            font=("Arial", 18),
            bg="#007D96",
            fg="white",
            width=5,
            height=3,
        )
        self.label.place(relx=x_rel, rely=y_rel, anchor="center")
        self.image_label = tk.Label(root)
        self.image_label.place(relx=x_rel + 0.06, rely=y_rel + 0.17, anchor="center")
        image = Image.open(image_path)
        tk_image = ImageTk.PhotoImage(image)

        self.default_image = tk_image
        self.image_label.config(height=200, width=317, image=self.default_image)

        self.rectangle_color = "white"
        self.rectangle_width = 80
        self.rectangle_height = 80
        self.rectangle_text = "Empty"

        self.rectangle_canvas_temp = tk.Canvas(
            root,
            width=self.rectangle_width,
            height=self.rectangle_height,
            bg=self.rectangle_color,
        )
        self.rectangle_canvas_temp.place(relx=x_rel + 0.12, rely=y_rel, anchor="center")
        self.body_temp_text = self.rectangle_canvas_temp.create_text(
            self.rectangle_width / 2,
            self.rectangle_height / 2 - 10,
            text="Body Temp",
            fill="black",
            font=("Arial", 10),
        )
        self.temp_text = self.rectangle_canvas_temp.create_text(
            self.rectangle_width / 2,
            self.rectangle_height / 2 + 10,
            text="98.4F",
            fill="black",
            font=("Arial", 14, "bold"),
        )

        self.rectangle_canvas_status = tk.Canvas(
            root,
            width=self.rectangle_width,
            height=self.rectangle_height,
            bg=self.rectangle_color,
        )
        self.rectangle_canvas_status.place(
            relx=x_rel + 0.06, rely=y_rel, anchor="center"
        )
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


class WebcamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Webcam Face Recognition")

        self.bg_image = PhotoImage(file=background)
        self.bg_width = 1920
        self.bg_height = 1200

        self.root.geometry(f"{self.bg_width}x{self.bg_height}")

        self.bg_label = tk.Label(root, image=self.bg_image)
        self.bg_label.place(relwidth=1, relheight=1)

        self.seats = [
            Seat(root, "A1", face_img, 0.66, 0.2),
            Seat(root, "A2", face_img, 0.26, 0.2),
            Seat(root, "B1", face_img, 0.66, 0.58),
            Seat(root, "B2", face_img, 0.26, 0.58),
        ]
        # single camera Seat coordinate
        # self.seat_coordinate = [(80, 975, 407, 501, "A1"),(1170, 976, 764, 630, "A2"),(408, 902, 649, 443, "B1"),(976, 749, 741, 463, "B2")]

        # two camera Seat coordinate
        self.seat_coordinate = [
            (15, 508, 320, 10, "A1"),
            (564, 508, 896, 10, "A2"),
            (920, 508, 1234, 5, "B1"),
            (1429, 508, 1767, 11, "B2"),
        ]

        self.database = get_passenger_data()
        self.vid = None
        self.frame = None
        self.monitoring_thread = None
        self.monitoring = False
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.frame_process = 0
        self.process_frame = 5

    def start_monitoring(self):
        for seat, passager in zip(self.seats, self.database):
            self.update_single_seat(seat, passager.get("passenger_image"))

        if not self.monitoring:
            self.monitoring = True
            messagebox.showinfo("Info", "Monitoring started.")

        self.start_webcam()

    def start_webcam(self):
        self.vid = create_webcam_stream(camera_source_1, camera_source_2)
        self.vid.start()
        self.show_frames()

    def seat_belt_process(self, frame):
        image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        image = (image / 127.5) - 1
        prediction = model.predict(image, verbose=0)
        index = np.argmax(prediction)
        label_name = class_names[index]
        confidence_score = prediction[0][index]
        return label_name, confidence_score

    def process_seat(self):
        seat_status = {}

        for x1, y1, x2, y2, seat_name in self.seat_coordinate:
            try:
                seat = self.frame[y2:y1, x1:x2]
                class_name, confidence_score = self.seat_belt_process(seat)
                label = f" {seat_name} ::  {class_name}: {round(confidence_score * 100, 2)}%"
                seat_status[seat_name] = seat_status[class_name]
            except Exception as e:
                print(e)

        return seat_status

    def show_frames(self, width=1800, height=900):

        try:
            if self.vid.stopped:
                return

            self.frame = self.vid.read()
            if self.frame is not None:
                self.frame_process += 1
                print(self.frame_process)
                self.frame = resize(self.frame, width, height)
                self.process_frames()
                _ , seatbelt_status = self.process_seat()
                self.display_frames(seatbelt_status)

                key = cv2.waitKey(1)
                if key == ord("q"):
                    self.vid.stop()
                    cv2.destroyAllWindows()
                    return

            self.root.after(100, self.show_frames)

        except Exception as e:
            logger.error("Error in frame processing: %s", e)

    def process_frames(self):
        dataset = {passenger["passenger_name"]: [passenger["passenger_name"],passenger["passenger_assign_seat"],passenger["passenger_encoding"]] for passenger in self.database }

        try:
            result = process_faces(self.frame, self.seat_coordinate)
            seatbelt_status = self.process_seat()


            for seat, (seat_name, face_embed) in zip(self.seats, result):

                if face_embed:
                    passenger_name, passenger_seat, _ = do_face_verification(dataset, face_embed)

                    seat_status = ("Correct" if passenger_seat == seat_name else "InCorrect")

                    if ( seat_status == "Correct" and seatbelt_status[seat_name] == "Seat Belt"):
                        color = "Green"
                        status = "Ready"
                    elif (seat_status == "Correct" and seatbelt_status[seat_name] == "No Seat Belt"):
                        color = "Yellow"
                        status = "correct"
                    elif seat_status == "InCorrect" and passenger_name != "Unknown":
                        color = "Orange"
                        status = "Incorrect"
                    elif passenger_name == "Unknown":
                        color = "Red"
                        status = "Unauthorized"

                    self.update_single_seat(seat, None, color, status)
                    logger.info(f"Verify passenger_name: {passenger_name}\nSeat Status: {seat_status}\nProcess seat frame {seat_name}")

                elif not face_embed:
                    self.update_single_seat(seat)

        except Exception as e:
            logger.error("Error in frame processing: %s", e)

    def display_frames(self, seat_status):

        draw_seats(self.frame, self.seat_coordinate)
        for (x1, y1, x2, y2, seat_name), status in zip(self.seat_coordinate, seat_status):
            color = (0, 255, 0) if status == "Seat Belt" else (0, 0, 255)
            cv2.putText(self.frame,f"Seat {status}",(x1, y1 - 30),font,0.5,color,2,lineType)
        cv2.imshow("Cabin monitoring", self.frame)

    def update_single_seat(self, seat, image_data=None, rectangle_color="white", status="Empty"):
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
    root = tk.Tk()
    app = WebcamApp(root)
    start_button = tk.Button(
        root,
        text="Start Monitoring",
        command=app.start_monitoring,
        font=("Arial", 18, "bold"),
        bg="#04AA6D",
        fg="white",
    )
    start_button.place(relx=0.5, rely=0.9, anchor="center")
    root.bind("<F1>", lambda event: root.attributes("-fullscreen", True))
    root.bind("<Escape>", lambda event: root.attributes("-fullscreen", False))
    root.mainloop()


if __name__ == "__main__":
    main()
