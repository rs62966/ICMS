import json
import pathlib
import time

import cv2
import mediapipe as mp

from CameraAccess import create_webcam_stream
from helper import (draw_seats, seats_coordinates,  # ,process_faces
                    time_consumer)
from log import Logger


class Config:
    """Configuration class for ICMS Dashboard."""

    def __init__(self):
        """Initialize configuration parameters."""
        current = pathlib.Path(__file__).parent.resolve()
        self.background = current.joinpath("Images", "home.png")

        with open(current.joinpath("config.json")) as data_file:
            data = json.load(data_file)

        self.camera_source_1 = data["CAMERA"]["FIRST_CAMERA_INDEX"]
        self.camera_source_2 = data["CAMERA"]["SECOND_CAMERA_INDEX"]
        self.seat_coordinates = seats_coordinates(data["SEAT_COORDINATES"], data["FRAME_SHAPE"])


def process_faces(frame, seat_coordinates, face_detection):
    """
    Process faces in the given frame and return a dictionary with seat information.
    """
    processed_seats = {}

    for x1, y1, x2, y2, seat_name in seat_coordinates:
        seat_roi = frame[y2:y1, x1:x2]

        # Resize the image for faster processing
        small_seat_roi = cv2.resize(seat_roi, (0, 0), fx=0.5, fy=0.5)

        # Convert the image to RGB
        rgb_seat_roi = cv2.cvtColor(small_seat_roi, cv2.COLOR_BGR2RGB)

        # Run face detection
        results = face_detection.process(rgb_seat_roi)

        # If a face is detected, get the face encoding
        if results.detections:
            face_detection_data = results.detections[0]
            bboxC = face_detection_data.location_data.relative_bounding_box
            ih, iw, _ = rgb_seat_roi.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            face_roi = rgb_seat_roi[y : y + h, x : x + w]

            # You can further process the face_roi if needed

            processed_seats[seat_name] = face_roi
        else:
            processed_seats[seat_name] = []

    return processed_seats


@time_consumer
def test_time(frame, seat_coordinates, mp_face_detection):
    processed_seats = process_faces(frame, seat_coordinates, mp_face_detection)
    logger.info(f"processed_seats {[{key:len(value)} for key , value in processed_seats.items()]}")
    # Display frames with seat ROI for seat mapping
    cv2.namedWindow("Cabin monitoring", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Cabin monitoring", 600, 300)
    draw_seats(frame, CONFIG.seat_coordinates)
    cv2.imshow("Cabin monitoring", frame)
    time.sleep(1)


if __name__ == "__main__":
    CONFIG = Config()
    logger = Logger(module="TEST Dashboard")
    try:
        mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
        vid = create_webcam_stream(CONFIG.camera_source_1, CONFIG.camera_source_2)
        vid.start()

        while True:
            if vid.stopped:
                break

            frame = vid.read()
            if frame is not None:
                # Process frames and store every seat face signature
                test_time(frame, CONFIG.seat_coordinates, mp_face_detection)
                # Check for the 'q' key to stop the video stream
                key = cv2.waitKey(1)
                if key == ord("q"):
                    vid.stop()
                    cv2.destroyAllWindows()
                    break
    except Exception as e:
        logger.error(f"Error in show_frames: {e}")
