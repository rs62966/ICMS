import threading
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QSpacerItem, QSizePolicy
from PyQt5.QtCore import Qt,QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QMovie, QIcon
from helper import CameraWidget, ObjectsFiles,Config, draw_seats
from detection_models import YoloObjectdetection,BehaviourDetection


class BoardThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    object_detected = pyqtSignal(list)
    
    def __init__(self, cap1):
        super().__init__()
        self.stopped = False
        self.cap1 = cap1

    def run(self):
        try:
            while not self.stopped:
                ret1, frame1 = self.cap1.read()
            
                if not ret1:
                    break

                frame1 = cv2.resize(frame1, (500, 480))
                
                detected_classes_list = YOLO_OBJECT.process_objects(frame1)
                self.object_detected.emit(detected_classes_list)
                
                self.frame_ready.emit(frame1)

        except Exception as e:
            print(e)
            
    def stop(self):
        self.stopped = True

class CabinThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    behaviour_detected = pyqtSignal(dict)

    def __init__(self, cap2):
        super().__init__()
        self.stopped = False
        self.cap2 = cap2

    def run(self):
        try:
            while not self.stopped:
                ret2, frame2 = self.cap2.read()

                if not ret2:
                    break

                frame2 = cv2.resize(frame2, (660, 480))

                behaviour_dict = YOLO_GESTURE.process_behaviour(frame2)
                self.behaviour_detected.emit(behaviour_dict)

                frame2 = draw_seats(frame2, CONFIG.seat_coordinates)
                self.frame_ready.emit(frame2)

        except Exception as e:
            print(e)
            
    def stop(self):
        self.stopped = True
              
class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        
        # Initialize VideoCapture instances outside threads
        cap1 = cv2.VideoCapture(CONFIG.camera_source_1)
        cap2 = cv2.VideoCapture(CONFIG.camera_source_2)

        # Create threads with initialized VideoCapture instances
        self.board_thread = BoardThread(cap1)
        self.cabin_thread = CabinThread(cap2)

        self.camera_widget = None

        self.setGeometry(100, 100, 1920, 1080)
        self.setWindowTitle("CyICMS")
        self.setStyleSheet("background-color: #000000;")

        self.logo_path = "Images/Logo.png"
        original_pixmap = QPixmap(self.logo_path)
        target_size = original_pixmap.size().scaled(300, 300, Qt.KeepAspectRatio)
        resized_pixmap = original_pixmap.scaled(target_size, Qt.KeepAspectRatio)
        self.logo_label = QLabel(self)
        self.logo_label.setPixmap(resized_pixmap)
        self.logo_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        gif_path = "Images/Face.gif"
        gif_label = QLabel(self)
        gif_movie = QMovie(gif_path)
        gif_label.setMovie(gif_movie)
        gif_movie.start()

        gif_label.setFixedSize(240, 240)
        gif_label.setScaledContents(True)

        left_coloumn = QWidget(self)
        left_coloumn.setStyleSheet("background-color: #171B2E;")

        left_layout = QVBoxLayout(left_coloumn)
        left_layout.addWidget(self.logo_label, alignment=Qt.AlignTop)

        spacer_item_top = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        left_layout.addItem(spacer_item_top)

        left_layout.addWidget(gif_label, alignment=Qt.AlignVCenter | Qt.AlignHCenter)

        spacer_item_between = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Fixed)
        left_layout.addItem(spacer_item_between)

        self.boardingmonitoring_button = QPushButton("Boarding Monitoring", self)
        self.boardingmonitoring_button.setStyleSheet("background-color: #3498db; color: #ffffff; font-weight: bold; font-size: 20px;")
        self.boardingmonitoring_button.setFixedSize(260, 40)
        left_layout.addWidget(self.boardingmonitoring_button, alignment=Qt.AlignVCenter | Qt.AlignHCenter)
        
        left_layout.addSpacing(20)
        
        self.cabinmonitoring_button = QPushButton("Cabin Monitoring", self)
        self.cabinmonitoring_button.setStyleSheet("background-color: #3498db; color: #ffffff; font-weight: bold; font-size: 20px;")
        self.cabinmonitoring_button.setFixedSize(260, 40)
        left_layout.addWidget(self.cabinmonitoring_button, alignment=Qt.AlignVCenter | Qt.AlignHCenter)

        spacer_item_bottom = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        left_layout.addItem(spacer_item_bottom)

        video_button = QPushButton(self)
        video_button.setIcon(QIcon("Icons/video.png"))
        video_button.setFixedSize(40, 40)

        speaker_button = QPushButton(self)
        speaker_button.setIcon(QIcon("Icons/speaker.png"))
        speaker_button.setFixedSize(40, 40)

        bottom_layout = QHBoxLayout()

        bottom_layout.addWidget(speaker_button, alignment=Qt.AlignBottom | Qt.AlignLeft)
        bottom_layout.addSpacing(200)
        bottom_layout.addWidget(video_button, alignment=Qt.AlignBottom | Qt.AlignRight)

        left_layout.addLayout(bottom_layout)

        right_layout = QVBoxLayout()

        text_label = QLabel("INTELLIGENT CABIN MANAGEMENT SYSTEM", self)
        text_label.setStyleSheet("color: #ffffff; font-size: 32px; font-weight: bold;")
        right_layout.addWidget(text_label, alignment=Qt.AlignTop | Qt.AlignRight)

        spacer_item = QSpacerItem(20, 30, QSizePolicy.Minimum, QSizePolicy.Expanding)
        right_layout.addItem(spacer_item)

        behaviour_rect_widget = QWidget(self)
        behaviour_rect_widget.setFixedSize(1480, 920)
        behaviour_rect_widget.setStyleSheet("background-color: #333333; border-radius: 10px;")

        behaviour_label = QLabel("BEHAVIOUR MONITORING", behaviour_rect_widget)
        behaviour_label.setStyleSheet("color: #00C8F0; font-size: 32px; font-weight: bold;")
        behaviour_label.setAlignment(Qt.AlignTop | Qt.AlignHCenter)

        behaviour_layout = QVBoxLayout(behaviour_rect_widget)
        behaviour_layout.addWidget(behaviour_label)
        heading_spacer = QSpacerItem(0, 60, QSizePolicy.Expanding, QSizePolicy.Minimum)
        behaviour_layout.addItem(heading_spacer)

        self.Empty_pixmap = QPixmap('Icons/Empty.png')
        self.Empty_pixmap = self.Empty_pixmap.scaled(120, 120, Qt.KeepAspectRatio)
        
        self.Aggressive_pixmap = QPixmap('Icons/angry.png')
        self.Aggressive_pixmap = self.Aggressive_pixmap.scaled(90, 90, Qt.KeepAspectRatio)
        
        self.Non_Aggressive_pixmap = QPixmap('Icons/smile.png')
        self.Non_Aggressive_pixmap = self.Non_Aggressive_pixmap.scaled(90, 90, Qt.KeepAspectRatio)

        def create_rectangle(label_text, pixmap, status_text, object_text):
            rectangle = QWidget()
            rectangle.setStyleSheet("background-color: #000000; border-radius: 5px;")
            rectangle.setFixedSize(400, 220)
            label = QLabel(label_text, rectangle)
            label.setStyleSheet("color: #00C8F0; font-size: 18px; font-weight: bold;")
            image_label = QLabel(rectangle)
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setPixmap(pixmap)
            status_text_label = QLabel(status_text, rectangle)
            status_text_label.setStyleSheet("color: white; font-weight: bold; font-size: 20px;")
            status_text_label.setAlignment(Qt.AlignCenter)
            object_text_label = QLabel(object_text, rectangle)
            object_text_label.setStyleSheet("color: white; font-weight: bold; font-size: 20px;")
            object_text_label.setAlignment(Qt.AlignCenter)
            
            rectangle.image_label = image_label
            rectangle.status_text_label = status_text_label
            rectangle.object_text_label = object_text_label
            
            layout = QVBoxLayout(rectangle)
            layout.addWidget(label, alignment=Qt.AlignTop | Qt.AlignHCenter)
            layout.addWidget(image_label, alignment=Qt.AlignCenter)
            layout.addWidget(status_text_label, alignment=Qt.AlignCenter)
            layout.addWidget(object_text_label, alignment=Qt.AlignCenter)
            layout.addStretch(1)
            rectangle.setLayout(layout)
            
            return rectangle

        horizontal_layout_A = QHBoxLayout()
        horizontal_layout_A.addSpacing(60)

        self.rectangle_A2 = create_rectangle("SEAT: A2", self.Empty_pixmap, "Status:Empty", "")
        self.rectangle_A1 = create_rectangle("SEAT: A1", self.Empty_pixmap, "Status:Empty", "")

        horizontal_layout_A.addWidget(self.rectangle_A2)
        horizontal_layout_A.addWidget(self.rectangle_A1)

        behaviour_layout.addLayout(horizontal_layout_A)
        layout_spacer = QSpacerItem(0, 100, QSizePolicy.Expanding, QSizePolicy.Minimum)
        behaviour_layout.addItem(layout_spacer)

        horizontal_layout_B = QHBoxLayout()
        horizontal_layout_B.addSpacing(60)

        self.rectangle_B2 = create_rectangle("SEAT: B2", self.Empty_pixmap, "Status:Empty", "")
        self.rectangle_B1 = create_rectangle("SEAT: B1", self.Empty_pixmap, "Status:Empty", "")

        horizontal_layout_B.addWidget(self.rectangle_B2)
        horizontal_layout_B.addWidget(self.rectangle_B1)

        behaviour_layout.addLayout(horizontal_layout_B)
        
        behaviour_layout.addSpacing(60)
        
        self.rectangles_dict = {'B1': self.rectangle_B1, 'B2': self.rectangle_B2}

        self.msg_rect = QLabel()
        self.msg_rect.setStyleSheet("background-color: #000000; border-radius: 5px;")
        self.msg_rect.setFixedSize(680, 220)
        self.heading = QLabel("OBJECT MONITORING")
        self.heading.setStyleSheet("color: #00C8F0; font-size: 26px; font-weight: bold;")

        self.object_msg = QLabel("No Objects Detected")
        self.object_msg.setStyleSheet("color: #ffffff; font-size: 26px; font-weight: bold;")
        self.object_msg.setWordWrap(True)
        self.object_msg.setAlignment(Qt.AlignCenter)
        self.object_msg.setFixedWidth(650)

        msg_layout = QVBoxLayout()
        msg_layout.addWidget(self.heading, alignment=Qt.AlignTop | Qt.AlignHCenter)
        msg_layout.addWidget(self.object_msg, alignment=Qt.AlignTop | Qt.AlignCenter)
        self.msg_rect.setLayout(msg_layout)
        behaviour_layout.addWidget(self.msg_rect, alignment=Qt.AlignHCenter)
        behaviour_layout.addSpacing(30)

        right_layout.addWidget(behaviour_rect_widget, alignment=Qt.AlignTop | Qt.AlignRight)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        main_layout = QHBoxLayout(self)
        main_layout.addWidget(left_coloumn, alignment=Qt.AlignLeft)
        main_layout.addWidget(right_widget, alignment=Qt.AlignBottom | Qt.AlignRight)

        self.setLayout(main_layout)

        self.cabinmonitoring_button.clicked.connect(self.start_cabinmonitoring)
        self.cabin_thread.frame_ready.connect(self.display_frame)
        self.cabin_thread.behaviour_detected.connect(self.update_behaviour_status)
        
        self.boardingmonitoring_button.clicked.connect(self.start_boardmonitoring)
        self.board_thread.frame_ready.connect(self.display_boardframe)
        self.board_thread.object_detected.connect(self.update_object_label)
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            self.stop_cabinmonitoring()
            self.stop_boardmonitoring()
        elif event.key() == Qt.Key_Space:
            self.start_boardmonitoring()
        elif event.key() == Qt.Key_Shift:
            self.start_cabinmonitoring()
            
    def display_frame(self, frame):
        if self.camera_widget is None or self.camera_widget.isHidden():
            self.camera_widget = CameraWidget(frame)
            self.camera_widget.closed.connect(self.stop_cabinmonitoring)
            self.camera_widget.show()
        else:
            self.camera_widget.update_frame(frame)

    def update_object_label(self, object_names):
        if isinstance(object_names, list):
            colors = []
            for obj in object_names:
                if obj in OBJECT.offensive_objects:
                    colors.append("red")
                else:
                    colors.append("white")
            
            colored_object_names = [f"<font color='{color}'>{obj}</font>" for obj, color in zip(object_names, colors)]
            object_names_str = ", ".join(colored_object_names)
            self.object_msg.setText(object_names_str if object_names_str != "" else "No Objects Detected")
            
        else:
            if object_names in OBJECT.offensive_objects:
                self.object_msg.setText(f"<font color='red'>{object_names}</font>")
            else:
                self.object_msg.setText(str(object_names) if str(object_names) != "" else "No Objects Detected")
                
    def update_behaviour_status(self, behaviour_dict):
        for key, rectangle in self.rectangles_dict.items():
            rectangle.status_text_label.setText(behaviour_dict[key])
            if behaviour_dict[key] == 'Aggressive':
                rectangle.status_text_label.setStyleSheet("color: Red; font-weight: bold; font-size: 20px;")
                rectangle.image_label.setPixmap(self.Aggressive_pixmap)
            elif behaviour_dict[key] == 'Non-Aggressive':
                rectangle.status_text_label.setStyleSheet("color: Green; font-weight: bold; font-size: 20px;")
                rectangle.image_label.setPixmap(self.Non_Aggressive_pixmap)
            else:
                rectangle.status_text_label.setStyleSheet("color: White; font-weight: bold; font-size: 20px;")
                rectangle.image_label.setPixmap(self.Empty_pixmap)
                
    def start_cabinmonitoring(self):
        self.cabinmonitoring_button.setEnabled(False)
        self.cabin_thread.start()
        
    def stop_cabinmonitoring(self):
        for key, rectangle in self.rectangles_dict.items():
            rectangle.status_text_label.setText("Empty")
            rectangle.status_text_label.setStyleSheet("color: White; font-weight: bold; font-size: 20px;")
            rectangle.image_label.setPixmap(self.Empty_pixmap)
            
        if self.cabin_thread.isRunning():
            self.cabin_thread.stopped = True
            self.cabin_thread.wait()
        if self.camera_widget:
            self.camera_widget.close()
            
    def display_boardframe(self, frame):
        if self.camera_widget is None or self.camera_widget.isHidden():
            self.camera_widget = CameraWidget(frame)
            self.camera_widget.closed.connect(self.stop_boardmonitoring)
            self.camera_widget.show()
        else:
            self.camera_widget.update_frame(frame)
            
    def start_boardmonitoring(self):
        self.boardingmonitoring_button.setEnabled(False)
        self.board_thread.start()
        
    def stop_boardmonitoring(self):
        self.object_msg.setText("No Objects Detected")
        if self.board_thread.isRunning():
            self.board_thread.stopped = True
            self.board_thread.wait()
        if self.camera_widget:
            self.camera_widget.close()
            
def ModelLoader():
    # Initialize necessary components
    global CONFIG, OBJECT, YOLO_OBJECT, YOLO_GESTURE
    CONFIG = Config()
    OBJECT = ObjectsFiles()
    YOLO_OBJECT = YoloObjectdetection()
    YOLO_GESTURE = BehaviourDetection()

if __name__ == "__main__":
    start = time.time()  # Start timing
    
    # Start a separate thread to initialize necessary components
    init_thread = threading.Thread(target=ModelLoader)
    init_thread.start()
    
    # Start GUI application
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    
    # Wait for the initialization thread to complete
    init_thread.join()
    
    # Calculate and print total time taken
    end = time.time()
    print("Total time taken by system:", (end - start))
    
    # Exit application loop
    sys.exit(app.exec_())