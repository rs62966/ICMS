import pyttsx3
import speech_recognition as sr

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

seat = "A1"
message = f"Seat {seat}, wrong passenger"

engine.say(message)
engine.runAndWait()