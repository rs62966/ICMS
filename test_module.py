from collections import defaultdict
from io import BytesIO
import os
from pprint import pprint
from timeit import default_timer as timer

from PIL import Image, ImageTk

from database import get_passenger_data
from helper import Seat, face_img, Logger

logger = Logger(module="Test Module")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
start = timer()

EMPTY = 0
CORRECT = 1
INCORRECT = -1
UNAUTHORIZED = 101


# Example usage with the provided data
last_five_frames = {
    1: {
        "A1": [{"passenger_name": "Uma Shanker", "passenger_assign_seat": "A1", "passenger_match_distance": 0.5113961492903852}],
        "A2": [{"passenger_name": "Adarsh", "passenger_assign_seat": "A2", "passenger_match_distance": 0.5113961492903852}],
        "B1": [{"passenger_name": "Ravi Shanker Singh", "passenger_assign_seat": "B1", "passenger_match_distance": 0.5113961492903852}],
        "B2": [{"passenger_name": "Hari", "passenger_assign_seat": "B2", "passenger_match_distance": 0.5113961492903852}],
    },
    2: {
        "A1": [{"passenger_name": "Uma Shanker", "passenger_assign_seat": "B2", "passenger_match_distance": 0.5113961492903852}],
        "A2": [{"passenger_name": "Adarsh", "passenger_assign_seat": "A2", "passenger_match_distance": 0.5113961492903852}],
        "B1": [{"passenger_name": "Ravi Shanker Singh", "passenger_assign_seat": "B1", "passenger_match_distance": 0.5113961492903852}],
        "B2": [{"passenger_name": "Hari", "passenger_assign_seat": "B2", "passenger_match_distance": 0.5113961492903852}],
    },
    3: {
        "A1": [{"passenger_name": "Uma Shanker", "passenger_assign_seat": "B2", "passenger_match_distance": 0.5113961492903852}],
        "A2": [{"passenger_name": "Adarsh", "passenger_assign_seat": "A2", "passenger_match_distance": 0.5113961492903852}],
        "B1": [{"passenger_name": "Ravi Shanker Singh", "passenger_assign_seat": "B1", "passenger_match_distance": 0.5113961492903852}],
        "B2": [{"passenger_name": "Hari", "passenger_assign_seat": "B2", "passenger_match_distance": 0.5113961492903852}],
    },
    4: {
        "A1": [{"passenger_name": "Uma Shanker", "passenger_assign_seat": "B2", "passenger_match_distance": 0.5113961492903852}],
        "A2": [{"passenger_name": "Adarsh", "passenger_assign_seat": "A2", "passenger_match_distance": 0.5113961492903852}],
        "B1": [{"passenger_name": "Ravi Shanker Singh", "passenger_assign_seat": "B1", "passenger_match_distance": 0.5113961492903852}],
        "B2": [{"passenger_name": "Hari", "passenger_assign_seat": "A1", "passenger_match_distance": 0.5113961492903852}],
    },
    5: {
        "A1": [{"passenger_name": "Uma Shanker", "passenger_assign_seat": "B2", "passenger_match_distance": 0.5113961492903852}],
        "A2": [{"passenger_name": "Adarsh", "passenger_assign_seat": "A2", "passenger_match_distance": 0.5113961492903852}],
        "B1": [{"passenger_name": "Ravi Shanker Singh", "passenger_assign_seat": "B1", "passenger_match_distance": 0.5113961492903852}],
        "B2": [{"passenger_name": "Hari", "passenger_assign_seat": "A1", "passenger_match_distance": 0.5113961492903852}],
    },
}


last_five_frames2 = {
    1: {
        "A1": [],
        "A2": [{"passenger_name": "Adarsh", "passenger_assign_seat": "A2", "passenger_match_distance": 0.5113961492903852}],
        "B1": [{"passenger_name": "Ravi Shanker Singh", "passenger_assign_seat": "B1", "passenger_match_distance": 0.5113961492903852}],
        "B2": [{"passenger_name": "Hari", "passenger_assign_seat": "B2", "passenger_match_distance": 0.5113961492903852}],
    },
    2: {
        "A1": [],
        "A2": [],
        "B1": [{"passenger_name": "Ravi Shanker Singh", "passenger_assign_seat": "B1", "passenger_match_distance": 0.5113961492903852}],
        "B2": [{"passenger_name": "Hari", "passenger_assign_seat": "B2", "passenger_match_distance": 0.5113961492903852}],
    },
    3: {
        "A1": [],
        "A2": [{"passenger_name": "Adarsh", "passenger_assign_seat": "A2", "passenger_match_distance": 0.5113961492903852}],
        "B1": [{"passenger_name": "Ravi Shanker Singh", "passenger_assign_seat": "B1", "passenger_match_distance": 0.5113961492903852}],
        "B2": [{"passenger_name": "Hari", "passenger_assign_seat": "B2", "passenger_match_distance": 0.5113961492903852}],
    },
    4: {
        "A1": [],
        "A2": [{"passenger_name": "Adarsh", "passenger_assign_seat": "A2", "passenger_match_distance": 0.5113961492903852}],
        "B1": [{"passenger_name": "Ravi Shanker Singh", "passenger_assign_seat": "B1", "passenger_match_distance": 0.5113961492903852}],
        "B2": [{"passenger_name": "Hari", "passenger_assign_seat": "A1", "passenger_match_distance": 0.5113961492903852}],
    },
    5: {
        "A1": [],
        "A2": [{"passenger_name": "Adarsh", "passenger_assign_seat": "A2", "passenger_match_distance": 0.5113961492903852}],
        "B1": [{"passenger_name": "Ravi Shanker Singh", "passenger_assign_seat": "B1", "passenger_match_distance": 0.5113961492903852}],
        "B2": [{"passenger_name": "Hari", "passenger_assign_seat": "A1", "passenger_match_distance": 0.5113961492903852}],
    },
}


class NotificationController:
    SEAT_NAMES = ["A1", "A2", "B1", "B2"]
    UNAUTHORIZED_NAMES = {"Unknown", "Un"}

    def __init__(self, root=None, frame_process=5, data_point=None):
        self.frame_process = frame_process
        self.dataset = data_point
        self.root = root
        self.seats = None
        self.seat_info = None

    def initialize_seats(self, root):
        seat_positions = [
            ("A1", face_img, 0.66, 0.2),
            ("A2", face_img, 0.26, 0.2),
            ("B1", face_img, 0.66, 0.58),
            ("B2", face_img, 0.26, 0.58),
        ]
        return {name: Seat(root, name, img, x, y) for name, img, x, y in seat_positions}

    def initialize_seat_info(self):
        self.seats = self.initialize_seats(self.root)
        default_seat_info = {
            "passenger_name": "",
            "status": "Empty",
            "color": "white",
            "profile_image": None,
            "passenger_embedding": None,
        }

        seat_info = {seat_name: default_seat_info.copy() for seat_name in self.SEAT_NAMES}
        if self.dataset:
            for passenger in self.dataset:
                passenger_name, passenger_seat, passenger_embedding = passenger["passenger_dataset"]
                seat_info[passenger_seat]["profile_image"] = passenger["passenger_image"]
                seat_info[passenger_seat]["passenger_name"] = passenger_name
                seat_info[passenger_seat]["passenger_embedding"] = passenger_embedding
                self.update_single_seat(self.seats[passenger_seat], image_data=passenger["passenger_image"])
        else:
            logger.error("Database not able to fetch Data.")

        return seat_info

    def update_seat_info(self, frame_results):
        self.passenger_track = defaultdict(int)

        for frame_no, four_seat_info in frame_results.items():
            for seat_name, passengers in four_seat_info.items():
                name, status, score = "", "Empty", EMPTY
                if passengers:
                    passenger_info = passengers[0]
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
                self.passenger_track[seat_name, frame_no] = (name, status, score)

        return self.passenger_track

    def analysis(self, frame_results):
        result = {}
        analysis_seat_info = self.update_seat_info(frame_results)
        passenger_track = self.analyze_frames(analysis_seat_info)

        # Create a copy of the dictionary to avoid "dictionary changed size during iteration" error
        passenger_track_copy = passenger_track.copy()

        for seat, passenger_info in passenger_track_copy.items():
            passenger_name = passenger_info.get("passenger_name", "")
            status = passenger_info.get("status", "Empty")
            count = passenger_info.get("score", 0)
            color = "white"

            if passenger_name:
                if count >= 2:
                    status = "Correct"
                    color = "Yellow"
                elif count > -5:
                    status = "Incorrect"
                    color = "Orange"

            elif any(passenger_track[seat]["score"] > 0 for seat in self.UNAUTHORIZED_NAMES):
                status = "Unauthorized"
                color = "Red"

            result[seat] = [passenger_name, status, color, count]

        return result

    def analyze_frames(self, passenger_track):
        seat_analysis = defaultdict(lambda: {"passenger_name": "", "status": "Empty", "score": 0})

        for seat_info, passenger_info in passenger_track.items():
            seat_name, frame_no = seat_info
            passenger_name, status, score = passenger_info

            seat_analysis[seat_name]["passenger_name"] = passenger_name
            seat_analysis[seat_name]["status"] = status
            seat_analysis[seat_name]["score"] += score

        return seat_analysis

    def update_single_seat(self, seat, image_data=None, rectangle_color="white", status="Empty"):
        if image_data:
            load_image = Image.open(BytesIO(image_data))
            tk_image = ImageTk.PhotoImage(load_image)
            seat.image_label.config(image=tk_image)
            seat.image_label.image = tk_image
        seat.change_rectangle_color(rectangle_color, status)


database = get_passenger_data()
notification_controller = NotificationController(data_point=database)
analysis = notification_controller.analysis(last_five_frames2)
pprint(analysis)

end = timer()
logger.info("Elapsed time: " + str(end - start))
