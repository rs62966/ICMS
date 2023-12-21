import logging
import face_recognition
import cv2
import numpy as np
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Access environment variables for database configuration
db_config = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_DATABASE"),
    "use_pure": os.getenv("DB_USE_PURE", False),  # Use pure Python implementation, default to True if not provided
}
camera_source_1  = os.getenv("FIRST_CAMERA_INDEX")
camera_source_2 = os.getenv("SECOND_CAMERA_INDEX")

def process_faces(frame, seat_coordinates):
    """
    Process faces in the given frame and return a list of seat information.
    """
    processed_seats = []
    
    for x1, y1, x2, y2, name in seat_coordinates:
        seat_roi = frame[y2:y1, x1:x2]
        rgb_seat_roi = cv2.cvtColor(seat_roi, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_seat_roi)
        if len(face_locations) == 1:
            face_encodings = face_recognition.face_encodings(rgb_seat_roi, face_locations)
            processed_seats.append((name, face_encodings))
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
