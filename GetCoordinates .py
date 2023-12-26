import json
import pathlib

import cv2

from helper import setup_logger

# Set up logging
logger = setup_logger()
current = pathlib.Path(__file__).parent.resolve()

with open(current.joinpath("config.json")) as data_file:
    data = json.load(data_file)
    seat_coordinates = [(*coords, seat_name) for seat_name, coords in data["SEAT_COORDINATES"].items()]


class RectangleDrawer:
    def __init__(self, image_path):
        self.img = cv2.imread(image_path)
        self.drawing = False
        self.top_left_pt, self.bottom_right_pt = (-1, -1), (-1, -1)
        cv2.namedWindow("Coordinate Create WIndow")
        cv2.setMouseCallback("Coordinate Create WIndow", self.draw_rectangle)

    def draw_rectangle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.top_left_pt = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.bottom_right_pt = (x, y)
            cv2.rectangle(self.img, self.top_left_pt, self.bottom_right_pt, (0, 255, 0), 2)
            x1, y2 = self.top_left_pt
            x2, y1 = self.bottom_right_pt
            logger.info(f"Rectangle Coordinates: {[x1, y1, x2, y2]}")
            cv2.imshow("Coordinate Create WIndow", self.img)

    def draw_existing_rectangles(self, rectangles):
        for rect in rectangles:
            x1, y1, x2, y2, name = rect
            cv2.rectangle(self.img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(self.img, f"Seat {name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    def reset_rectangles(self):
        self.img = cv2.imread(image_path)
        self.top_left_pt, self.bottom_right_pt = (-1, -1), (-1, -1)

    def start(self):
        while True:
            cv2.imshow("Coordinate Create WIndow", self.img)
            key = cv2.waitKey(1) & 0xFF

            # Press 'r' to reset the drawn rectangle
            if key == ord("r"):
                self.reset_rectangles()

            # Press 'q' to exit the loop
            elif key == ord("q"):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = r"C:\Users\rs62966\Downloads\MicrosoftTeams-image (5).png"

    # Example usage with predefined rectangles
    predefined_rectangles = seat_coordinates

    app = RectangleDrawer(image_path)
    # app.draw_existing_rectangles(predefined_rectangles)
    app.start()
