import os
import cv2
import numpy as np
from CameraAccess import create_webcam_stream
from helper import resize, draw_seats
import time
from keras.models import load_model
from face_recognition import face_locations

model = load_model("keras_model.h5", compile=False)
class_names = ["Seat Belt", "No Seat Belt"]

# fmt: off
def seat_belt_process(frame):
    image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    image = (image / 127.5) - 1
    prediction = model.predict(image.reshape(1, 224, 224, 3), verbose=0)
    index = np.argmax(prediction)
    label_name = class_names[index]
    confidence_score = prediction[0][index]
    return label_name, confidence_score

# fmt: off
def process_seatbelt(frame, seat_coordinate):
    for x1, y1, x2, y2, seat_name in seat_coordinate:
        try:
            seat = frame[y2:y1, x1:x2]
            class_name, confidence_score = seat_belt_process(seat)
            label = f" {seat_name} ::  {class_name}: {round(confidence_score * 100, 2)}%"
            cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        except Exception as e:
            print(e)

    return frame

# fmt: off
def process_faces(frame, seat_coordinates):
    for (x1, y1, x2, y2, name) in seat_coordinates:
        seat_roi = frame[y2:y1, x1:x2]
        face_areas = face_locations(seat_roi)

        for face_location in face_areas:
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (x1 + left, y2 + top), (x1 + right, y2 + bottom), (255, 0, 0), 2)
            cv2.putText(frame, f"Face in Seat {name}", (x1 + left, y2 + top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)

    return frame

# fmt: off
def main(face_detection=True, seat_belt_detection=False):
    camera_source_1 = os.getenv("FIRST_CAMERA_INDEX")
    camera_source_2 = os.getenv("SECOND_CAMERA_INDEX")
    width, height = 1800, 900
    frame_time = 1
    frame_process = 0
    seat_coordinates = [
        (15, 508, 320, 10, "A1"),
        (564, 508, 896, 10, "A2"),
        (920, 508, 1234, 5, "B1"),
        (1429, 508, 1767, 11, "B2"),
    ]
    video_capture = create_webcam_stream(int(camera_source_1), int(camera_source_2))
    video_capture.start()

    prev_frame_time = 0

    try:
        if video_capture.stopped:
            print("[Exiting]: Error accessing webcam stream.")
            return
        while not video_capture.stopped:
            frame = video_capture.read()
            if frame is not None:
                frame_process += 1
                print(frame_process)
                frame = resize(frame, width, height)
                frame = draw_seats(frame, seat_coordinates)
                if seat_belt_detection:
                    frame = process_seatbelt(frame, seat_coordinates)
                if face_detection:
                    frame = process_faces(frame, seat_coordinates)

                new_frame_time = time.time()
                fps = 1 / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time

                fps_text = f"FPS: {fps:.2f}"
                cv2.putText(frame, fps_text, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

                cv2.imshow("Video", frame)

            if cv2.waitKey(frame_time) & 0xFF == ord("q"):
                video_capture.stop()
                break

    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
