import cv2
from helper import ObjectsFiles,Config
import numpy as np
from keras.models import load_model
import torch
from pathlib import Path

OBJECT = ObjectsFiles()
CONFIG = Config()
behaviour_model = load_model("BehaviourModel/behaviour.h5", compile=False)
behaviour_class = ['Non-Aggressive', 'Aggressive', 'Empty']

model_path = Path(OBJECT.yolov5_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
obj_model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path).to(device)
obj_model.eval()

class GunDetection:
    def __init__(self):
        self.smoothing_window_size = 15
        self.gun_history = []

    def detect_gun(self, frame):
        gun_cascade = cv2.CascadeClassifier('GunModel/gun.xml')
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        guns = gun_cascade.detectMultiScale(frame_gray, scaleFactor=1.3, minNeighbors=20, minSize=(100, 100))

        if len(guns) > 0:
            return 'pistol'
        else:
            return 'No pistol'

    def process_gun_detection(self, frame):
        gun_status = self.detect_gun(frame)
        self.update_history(gun_status)
        smoothed_gun_status = self.smooth_detection()
        return smoothed_gun_status

    def update_history(self, gun_status):
        self.gun_history.append(gun_status)
        if len(self.gun_history) > self.smoothing_window_size:
            self.gun_history.pop(0)

    def smooth_detection(self):
        if not self.gun_history:
            return 'No pistol'
        smoothed_status = max(set(self.gun_history), key=self.gun_history.count)
        return smoothed_status

class YoloObjectdetection:
    def __init__(self):
        self.smoothing_window_size = 15
        self.detected_classes_history = []
        self.gun_detector = GunDetection()
        
    def process_objects(self, frame):
        results = obj_model(frame)

        detected_classes = set()
        for obj in results.xyxy[0]:
            class_id = int(obj[5])
            class_name = obj_model.names[class_id]
            if class_name not in OBJECT.filter:
                detected_classes.add(class_name)
            gun = self.gun_detector.process_gun_detection(frame)
            if gun == 'pistol':
                detected_classes.add(gun)

        detected_classes_list = list(detected_classes)
        self.update_history(detected_classes_list)
        smoothed_classes = self.smooth_detection()
        return smoothed_classes

    def update_history(self, detected_classes_list):
        self.detected_classes_history.append(detected_classes_list)
        if len(self.detected_classes_history) > self.smoothing_window_size:
            self.detected_classes_history.pop(0)

    def smooth_detection(self):
        if not self.detected_classes_history:
            return []
        smoothed_classes = set()
        for classes_list in self.detected_classes_history:
            for class_name in classes_list:
                smoothed_classes.add(class_name)
        return list(smoothed_classes)
    
class BehaviourDetection:
    def __init__(self):
        self.seat_coordinate = CONFIG.seat_coordinates
        self.smoothing_window_size = 15
        self.behaviour_history = {}

    def behaviour_process(self, seatframe):
        image = cv2.resize(seatframe, (224, 224), interpolation=cv2.INTER_AREA)
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        image = (image / 127.5) - 1
        prediction = behaviour_model.predict(image, verbose=0)
        index = np.argmax(prediction)
        label_name = behaviour_class[index]
        confidence_score = prediction[0][index]
        return label_name, confidence_score

    def process_behaviour(self, frame):
        seat_behaviour_dict = {}
        for x1, y1, x2, y2, seat_name in self.seat_coordinate:
            try:
                seat = frame[y2:y1, x1:x2]
                class_name, confidence_score = self.behaviour_process(seat)
                confidence = round(confidence_score * 100, 2)
                self.update_history(seat_name, class_name, confidence)
                smoothed_class, smoothed_confidence = self.smooth_detection(seat_name)
                seat_behaviour_dict[seat_name] = smoothed_class
            except Exception as e:
                print(e)
        
        return seat_behaviour_dict

    def update_history(self, seat_name, class_name, confidence):
        if seat_name not in self.behaviour_history:
            self.behaviour_history[seat_name] = {'classes': [], 'confidences': []}

        history = self.behaviour_history[seat_name]
        history['classes'].append(class_name)
        history['confidences'].append(confidence)

        if len(history['classes']) > self.smoothing_window_size:
            history['classes'].pop(0)
            history['confidences'].pop(0)

    def smooth_detection(self, seat_name):
        history = self.behaviour_history.get(seat_name, {'classes': [], 'confidences': []})
        if not history['classes']:
            return None, None

        smoothed_class = max(set(history['classes']), key=history['classes'].count)
        smoothed_confidence = np.mean(history['confidences'])
        return smoothed_class, smoothed_confidence
    
class SeatObjects:
    def __init__(self):
        self.seat_coordinate = CONFIG.seat_coordinates
        self.smoothing_window_size = 15
        self.seat_obj_history = {}
        self.gun_detector = GunDetection()

    def process_objects(self, seatframe):
        results = obj_model(seatframe)

        detected_classes = set()
        for obj in results.xyxy[0]:
            class_id = int(obj[5])
            class_name = obj_model.names[class_id]
            if class_name not in OBJECT.filter:
                detected_classes.add(class_name)
            gun = self.gun_detector.process_gun_detection(seatframe)
            if gun == 'pistol':
                detected_classes.add(gun)

        detected_classes_list = list(detected_classes)
        return detected_classes_list

    def process_seatobjects(self, frame):
        seat_obj_dict = {}
        for x1, y1, x2, y2, seat_name in self.seat_coordinate:
            try:
                seat = frame[y2:y1, x1:x2]
                class_names = self.process_objects(seat)
                self.update_history(seat_name, class_names)
                smoothed_class_names = self.smooth_detection(seat_name)
                seat_obj_dict[seat_name] = smoothed_class_names
            except Exception as e:
                print(e)
                
        return seat_obj_dict

    def update_history(self, seat_name, class_names):
        if seat_name not in self.seat_obj_history:
            self.seat_obj_history[seat_name] = []

        history = self.seat_obj_history[seat_name]
        history.append(class_names)

        if len(history) > self.smoothing_window_size:
            history.pop(0)

    def smooth_detection(self, seat_name):
        history = self.seat_obj_history.get(seat_name, [])
        if not history:
            return []

        smoothed_classes = []
        for class_names in history:
            smoothed_classes.extend(class_names)

        unique_classes = set(smoothed_classes)
        return list(unique_classes)