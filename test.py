import customtkinter as tk
from PIL import Image, ImageTk


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


def create_engine():
    r = sr.Recognizer()
    engine = pyttsx3.init()
    engine.setProperty("rate", 100)
    try:
        engine.setProperty("voice", voices[22].id)
    except Exception as e:
        engine.setProperty("voice", voices[0].id)
    return engine, r


def speak(audio):
    engine.say(audio)
    engine.runAndWait()


name = "Ravi"
message = f"Dear {name} Welcome to Onboard"
voices = engine.getProperty("voices")
# for voice in voices:
#     engine.setProperty("voice", voice.id)
#     print(voice.id, '-->', voice.name)
#     speak(message)
#     time.sleep(2)


test = {"Ravi": True, "Rahul": True}

if name not in test.keys():
    print("Work")
else:
    print("not work")
