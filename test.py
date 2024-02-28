from collections import defaultdict
import customtkinter as tk
from PIL import Image, ImageTk

from helper import time_consumer


class Seat:
    def __init__(self, root, label, image_path, x_rel, y_rel):
        self.seat_name = label
        self.label = tk.CTkLabel(root, text=label, font=("Arial", 18), fg_color="white", width=5, height=3)
        self.label.place(relx=x_rel, rely=y_rel, anchor="center")
        self.image_label = tk.CTkLabel(root)
        self.image_label.place(relx=x_rel + 0.06, rely=y_rel + 0.17, anchor="center")
        image = Image.open(image_path)
        tk_image = ImageTk.PhotoImage(image)

        self.default_image = tk_image
        self.image_label.configure(height=200, width=317, image=self.default_image)

        self.rectangle_color = "white"
        self.rectangle_width = 200
        self.rectangle_height = 80
        self.rectangle_text = "Empty"

    #     self.rectangle_canvas_status = tk.CTkLabel(root, width=self.rectangle_width, height=self.rectangle_height, fg_color=self.rectangle_color)
    #     self.rectangle_canvas_status.place(relx=x_rel + 0.09, rely=y_rel, anchor="center")
    #     self.status_text = self.rectangle_canvas_status.create_text(
    #         self.rectangle_width / 2,
    #         self.rectangle_height / 2,
    #         text=self.rectangle_text,
    #         fill="black",
    #         font=("Arial", 12, "bold"),
    #     )

    # def change_rectangle_color(self, new_color, status):
    #     self.rectangle_canvas_status.config(fg_color=new_color)
    #     self.rectangle_text = status
    #     self.rectangle_canvas_status.itemconfig(self.status_text, text=status)


class WebcamApp(tk.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("1920x1200")
        self.title("ICMS Dashboard")
        # Set up GUI elements
        self.bg_image = ImageTk.PhotoImage(file=CONFIG.background)
        self.bg_label = tk.CTkLabel(self, image=self.bg_image)
        self.bg_label.place(relwidth=1, relheight=1)

        self.initialize_seats()

    def initialize_seats(self):
        seat_positions = [
            ("A1", face_img, 0.66, 0.2),
            ("A2", face_img, 0.26, 0.2),
            ("B1", face_img, 0.66, 0.58),
            ("B2", face_img, 0.26, 0.58),
        ]
        seats = {name: Seat(self, name, img, x, y) for name, img, x, y in seat_positions}
        return seats


# face_img = current.joinpath("Images", "face_icon.png")
# CONFIG = Config()
# app = WebcamApp()
# app.mainloop()

import pyttsx3
import speech_recognition as sr
import time

engine = pyttsx3.init()
voices = engine.getProperty("voices")

name = "Ravi Shanker Singh"
message = f"Dear {name} Welcome to Onboard"

def create_engine():
    r = sr.Recognizer()
    engine = pyttsx3.init()
    engine.setProperty("rate", 125)
    voices = engine.getProperty('voices')
    # For Codec USB Sound Card in set Persian voice tone 22, hindi 29 or english 12 
    try:
        engine.setProperty("voice", voices[22].id)
    except Exception as e:
        engine.setProperty("voice", voices[0].id)
    return engine, r
class WebcamApp:
    """Main class for the ICMS Dashboard application."""

    def __init__(self, root):
        self.track_last_five_frames = {}
        self.empty_skip_update_notification = 5
        self.ui_statbility = {"A1": 0, "A2": 0, "B1": 0, "B2": 0}
        self.welcome_notification = {}
    
    def update_gui(self):
        """Update the GUI based on seatbelt status."""
        empty_skip_notification = self.empty_skip_update_notification

        for seat, (name, status, color) in self.track_last_five_frames.items():
            if status == "Empty":
                self.ui_statbility[seat] += 1  # Initialize to 0 if it's None
                for key, value in self.ui_statbility.items():
                    if value == empty_skip_notification:
                        self.notification_controller.update_single_seat(seat, None, color, status)
                        self.ui_statbility[key] = 0
            else:
                self.notification_controller.update_single_seat(seat, None, color, status)
                weclome_message = f"Dear {name}, Welcome to onboard"
                incorrect_message = f"On {seat}, Wrong Passenger"
                unauthorize_message = f"Unauthorize Access"

                if color in ['yellow', 'green'] and name not in self.welcome_notification.keys():
                    self.speak(weclome_message)
                    self.welcome_notification[name] = True
                elif color == 'orange':
                    self.speak(incorrect_message)
                elif color == 'red':
                    self.speak(unauthorize_message)
        self.clear_frames()

