from collections import defaultdict
from copy import deepcopy
from io import BytesIO
from pprint import pprint
from timeit import default_timer as timer

from PIL import Image, ImageTk

from database import get_passenger_data
from helper import Seat, face_img, logger

# Start timer
start = timer()


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


class NotificationController:
    SEAT_NAMES = ["A1", "A2", "B1", "B2"]
    UNAUTHORIZED_NAMES = {"Unknown", "Un"}

    def __init__(self, root=None, frame_process=5, data_point=None):
        self.frame_process = frame_process
        self.dataset = data_point
        self.seats = None
        self.root = None
        self.seat_info = None
        self.passenger_track = defaultdict(int)

    def initialize_seats(self, root):
        seat_positions = [
            ("A1", face_img, 0.66, 0.2),
            ("A2", face_img, 0.26, 0.2),
            ("B1", face_img, 0.66, 0.58),
            ("B2", face_img, 0.26, 0.58),
        ]
        return {name: Seat(root, name, img, x, y) for name, img, x, y in seat_positions}

    def initialize_seat_info(self):
        """
        Initialize seat information based on the dataset.
        """
        self.seats = self.initialize_seats(self.root)
        default_seat_info = {
            "passenger_name": "",
            "status": "Empty",
            "color": "white",
            "profile_image": None,
            "passenger_embedding": None,
        }

        seat_info = {seat_name: deepcopy(default_seat_info) for seat_name in self.SEAT_NAMES}
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
        """
        Update seat information based on the frame results.
        """
        for frame_no, passengers in frame_results.items():
            for seat_name, passenger in passengers.items():
                if passenger:
                    passenger_info = passenger[0]
                    name, status = passenger_info.get("passenger_name", ""), "Unauthorized"

                    if name not in self.UNAUTHORIZED_NAMES:
                        seat_info = self.seat_info[seat_name]
                        seat_info["passenger_name"] = name
                        if passenger_info["passenger_assign_seat"] == seat_name:
                            status = "Correct"
                            self.passenger_track[(name, frame_no)] += 1
                        else:
                            status = "Incorrect"
                            self.passenger_track[(name, frame_no)] -= 1

                        seat_info["status"] = status

    def analysis(self):
        """
        Analyze seat information and determine the status of passengers.
        """
        result = {}
        passenger_track = self.get_passenger_last_seen_frame()
        for seat, passenger in self.seat_info.items():
            if passenger.get("passenger_name"):
                if passenger_track.get(passenger["passenger_name"], 0) >= 2:
                    passenger["status"] = "Correct"
                    passenger["color"] = "Yellow"
                else:
                    passenger["status"] = "Incorrect"
                    passenger["color"] = "Orange"
            elif passenger_track.get(passenger.get("passenger_name"), 0) > 0:
                passenger["status"] = "Unauthorized"
                passenger["color"] = "Red"
            result[seat] = [
                passenger.get("passenger_name"),
                passenger["status"],
                passenger["color"],
                passenger_track.get(passenger.get("passenger_name"), 0),
            ]

        return result

    def get_passenger_last_seen_frame(self):
        """
        Analyze persons with frames record for each passenger.
        """
        return {
            passenger_info["passenger_name"]: sum(
                self.passenger_track.get((passenger_info["passenger_name"], frame_no), 0)
                for frame_no in range(1, self.frame_process + 1)
                if self.passenger_track.get((passenger_info["passenger_name"], frame_no), 0) > 0
            )
            for seat_info in self.seat_info.values()
            if seat_info["status"] != "Empty"
            for passenger_info in [seat_info]
        }

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


# # Create an instance of NotificationController
# database = get_passenger_data()

# notification_controller = NotificationController(data_point=database)
# seat_info = notification_controller.initialize_seat_info()
# print(seat_info)

# # Assuming frame_results is a sample frame result dictionary
# # Update seat information and track frames
# # notification_controller.update_seat_info(last_five_frames)

# # Get the current seat information
# # analysis = notification_controller.analysis()

# # pprint(analysis)

# # Stop timer and print elapsed time
# end = timer()
# logger.info("Elapsed time: " + str(end - start))
