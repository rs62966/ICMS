import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from io import BytesIO
import pathlib
import sys
import tkinter as tk
from tkinter import PhotoImage, messagebox
import cv2
from PIL import Image, ImageTk
from CameraAccess import create_webcam_stream
from database import get_passenger_data
from helper import draw_seats,do_face_verification,process_faces, resize,logger
import numpy as np
from keras.models import load_model

current = pathlib.Path(__file__).parent.resolve()
EMBED_N = 128
face_img = current.joinpath("Images", "face_icon.png")
background = current.joinpath("Images", "home.png")
source = 0
source1 = 1
np.set_printoptions(suppress=True)

model = load_model("keras_model.h5", compile=False)

class_names = ['Seat Belt', 'No Seat Belt']

class Seat:
    def __init__(self, root, label, image_path, x_rel, y_rel):
        self.seat_name = label
        self.label = tk.Label(root, text=label, font=("Arial", 18), bg="#007D96", fg="white", width=5, height=3)
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

        self.rectangle_canvas_temp = tk.Canvas(root, width=self.rectangle_width, height=self.rectangle_height, bg=self.rectangle_color)
        self.rectangle_canvas_temp.place(relx=x_rel + 0.12, rely=y_rel, anchor="center")
        self.body_temp_text = self.rectangle_canvas_temp.create_text(self.rectangle_width / 2, self.rectangle_height / 2 - 10, text="Body Temp", fill="black", font=("Arial", 10))
        self.temp_text = self.rectangle_canvas_temp.create_text(self.rectangle_width / 2, self.rectangle_height / 2 + 10, text="98.4F", fill="black", font=("Arial", 14,'bold'))

        self.rectangle_canvas_status = tk.Canvas(root, width=self.rectangle_width, height=self.rectangle_height, bg=self.rectangle_color)
        self.rectangle_canvas_status.place(relx=x_rel + 0.06, rely=y_rel, anchor="center")
        self.status_text = self.rectangle_canvas_status.create_text(
            self.rectangle_width / 2,
            self.rectangle_height / 2,
            text=self.rectangle_text,
            fill="black",
            font=("Arial", 8,'bold'),
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
        self.seat_coordinate = [(80, 975, 407, 501, "A1"),(1170, 976, 764, 630, "A2"),(408, 902, 649, 443, "B1"),(976, 749, 741, 463, "B2")]
        
        # two camera Seat coordinate
        # self.seat_coordinate = [(15, 508, 320, 10, "A1"),(564, 508, 896, 10, "A2"),(920, 508, 1234, 5, "B1"),(1429, 508, 1767, 11, "B2")]
        
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
        self.vid = create_webcam_stream(source1)
        self.vid.start()
        self.show_frames()
        
    def seat_belt_process(self,frame):
        image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        image = (image / 127.5) - 1
        prediction = model.predict(image, verbose=0)
        index = np.argmax(prediction)
        label_name = class_names[index]
        confidence_score = prediction[0][index]
        return label_name, confidence_score

    def process_seat(self):
        seat_status = []
        seat_no = []
        for x1, y1, x2, y2, seat_name in self.seat_coordinate:
            try:
                seat = self.frame[y2:y1, x1:x2]
                class_name, confidence_score = self.seat_belt_process(seat)
                label = f" {seat_name} ::  {class_name}: {round(confidence_score * 100, 2)}%"
                seat_status.append(class_name)
                seat_no.append(seat_name)
            except Exception as e:
                print(e)
        
        return seat_no,seat_status
       
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
        dataset = {passenger["passenger_name"]: [passenger["passenger_name"], passenger["passenger_assign_seat"], passenger["passenger_encoding"]] for passenger in self.database}
        seat_status_dict = {}

        try:
            result = process_faces(self.frame, self.seat_coordinate)

            for seat, (seat_name, face_embed) in zip(self.seats, result):
                seat_no, seatbelt_status = self.process_seat()

                if face_embed:
                    passenger_name, passenger_seat, _ = do_face_verification(dataset, face_embed)

                    check_belt = [seatbelt_status[i] for i in range(4) if passenger_seat in [f"A{i+1}", f"B{i+1}", "Un"] and seat_no[i] not in seat_status_dict]

                    if check_belt:
                        seat_status_key = seat_no.pop()
                        seat_status_value = ', '.join(check_belt)
                        seat_status_dict[seat_status_key] = seat_status_value

                    seat_status = "Correct" if passenger_seat == seat_name else "InCorrect"

                    if seat_status == "Correct" and seat_status_dict[passenger_seat] == "Seat Belt":
                        color = "Green"
                        status = "Ready"
                    elif seat_status == "Correct" and seat_status_dict[passenger_seat] == "No Seat Belt":
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
            color = (0, 255, 0) if status == 'Seat Belt' else (0, 0, 255)
            cv2.putText(self.frame, f"Seat {status}",(x1, y1 - 30),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2,cv2.LINE_AA,)
            
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
    start_button = tk.Button(root, text="Start Monitoring", command=app.start_monitoring, font=("Arial", 18,'bold'), bg="#04AA6D", fg="white")
    start_button.place(relx=0.5, rely=0.9, anchor="center")
    root.bind("<F1>", lambda event: root.attributes("-fullscreen", True))
    root.bind("<Escape>", lambda event: root.attributes("-fullscreen", False))
    root.mainloop()

if __name__ == "__main__":
    main()