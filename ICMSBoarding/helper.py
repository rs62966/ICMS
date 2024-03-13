import cv2
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt,pyqtSignal
from PyQt5.QtGui import QPixmap,QImage
import pathlib
import json

class Config:
    def __init__(self):
        current = pathlib.Path(__file__).parent.resolve()
        self.background = current.joinpath("Images", "home.png")

        with open(current.joinpath("config.json")) as data_file:
            data = json.load(data_file)

        self.camera_source_1 = data["CAMERA"]["FIRST_CAMERA_INDEX"]
        self.camera_source_2 = data["CAMERA"]["SECOND_CAMERA_INDEX"]
        self.seat_coordinates = seats_coordinates(data["SEAT_COORDINATES"], data["FRAME_SHAPE"])

# fmt: off
class ObjectsFiles:
    def __init__(self):
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
            'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window',
            'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush', 'hair brush'
        ]

        self.filter = [
            "dining table", "window", "desk", "toilet", "door", "tv", "microwave", "oven", "toaster", "sink",
            "refrigerator", "chair", "couch", "potted plant", "bed", "kite", "cow", "elephant", "bear", "zebra",
            "giraffe", "horse", "sheep", "boat", "traffic light", "fire hydrant", "street sign", "stop sign",
            "parking meter", "bench", "car", "motorcycle", "airplane", "bus", "train", "truck", "person", "bicycle",
            "tie", "bird", "cat", "dog", "pizza", "teddy bear", "vase", "frisbee", "toothbrush", "donut", "cake",
            "bowl", "banana", "snowboard", "skateboard", "baseball glove", "sports ball"
        ]

        self.offensive_objects = ["scissors", "knife", "baseball bat", "fork", "pistol"]

        self.yolov5_path = 'Model/yolov5s.pt'

        
class CameraWidget(QWidget):
    closed = pyqtSignal()

    def __init__(self, frame=None, parent=None):
        super().__init__(parent)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        self.setLayout(layout)
        if frame is not None:
            self.update_frame(frame)

    def update_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            self.close()
            
def seats_coordinates(data, frame_shape):
    h, w, d = frame_shape
    return [(int(coord[0] * w), int(coord[1] * h), int(coord[2] * w), int(coord[3] * h), seat_name) for seat_name, coord in data.items()]
            
def draw_seats(frame, seat_coordinates):
    color = (0, 0, 255)

    for x1, y1, x2, y2, name in seat_coordinates:
        cv2.rectangle(frame, (x1, y2), (x2, y1), color, 5)
        cv2.putText(frame, f"Seat {name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    return frame