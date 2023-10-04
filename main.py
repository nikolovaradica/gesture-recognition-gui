import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import tkinter as tk
from PIL import Image, ImageTk
import pyautogui
import time


gesture_meanings = {"stop": "Start the video", "thumbs up": "Volume up", "thumbs down": "Volume down",
                    "okay": "Fast forward", "rock": "Rewind", "peace": "Play next video",
                    "live long": "Play previous video"}
keyboard_meanings = {"space": "Start the video", "up": "Volume up", "down": "Volume down",
                     "left": "Rewind", "right": "Fast forward", ("shift", "n"): "Play next video",
                     ("shift", "p"): "Play previous video", "none": "Do nothing"}


class StartWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Gesture Recognition")

        background_image = Image.open("icons/background.jpg")
        background_image = ImageTk.PhotoImage(background_image)
        self.background_label = tk.Label(root, image=background_image)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
        self.background_label.image = background_image

        self.start_button = tk.Button(self.root, text="LAUNCH", font=("Montserrat", 10), fg="#ffffff", bg="#8da4d5",
                                      command=self.launch_action, padx=20, pady=10)
        self.start_button.pack(pady=(120, 40), padx=250)

        self.settings_button = tk.Button(self.root, text="SETTINGS", font=("Montserrat", 10), fg="#ffffff",
                                         bg="#8da4d5", command=self.settings_action, padx=20, pady=10)
        self.settings_button.pack(pady=(40, 40), padx=250)

        self.exit_button = tk.Button(self.root, text="EXIT", font=("Montserrat", 10), fg="#ffffff", bg="#8da4d5",
                                     command=self.root.destroy, padx=20, pady=10)
        self.exit_button.pack(pady=(40, 120), padx=250)

    def launch_action(self):
        self.root.destroy()
        root = tk.Tk()
        main = MainWindow(root)
        main.run()

    def settings_action(self):
        self.root.destroy()
        root = tk.Tk()
        settings = SettingsWindow(root)
        settings.run()

    def run(self):
        self.root.mainloop()


class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Gesture Recognition")

        background_image = Image.open("icons/background.jpg")
        background_image = ImageTk.PhotoImage(background_image)
        self.background_label = tk.Label(root, image=background_image)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
        self.background_label.image = background_image

        self.settings_button = tk.Button(self.root, text="SETTINGS", command=self.open_settings_window,
                                         fg="#ffffff", bg="#8da4d5", pady=10, padx=10, font=("Montserrat", 10))
        self.settings_button.grid(row=0, column=1, padx=10, pady=10)

        self.back_button = tk.Button(self.root, text="BACK", command=self.open_start_window,
                                     fg="#ffffff", bg="#8da4d5", pady=10, padx=10, font=("Montserrat", 10))
        self.back_button.grid(row=0, column=2, padx=10, pady=10)

        self.camera = cv2.VideoCapture(0)
        self.camera_label = tk.Label(root, bg="#c9d4e8")
        self.camera_label.grid(row=1, column=1, columnspan=2)

        self.gesture_active = False
        self.last_gesture_time = time.time() + 3

        self.model = tf.keras.models.load_model('mp_hand_gesture')
        f = open('gestures.txt', 'r')
        self.gesture_names = f.read().split('\n')
        f.close()

        self.mph = mp.solutions.hands
        self.hands = self.mph.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.7)
        self.draw = mp.solutions.drawing_utils

        self.update()

    def update(self):
        ret, frame = self.camera.read()
        height, width, _ = frame.shape
        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            landmarks = []

            for mhl in result.multi_hand_landmarks:
                for l in mhl.landmark:
                    lx, ly = int(l.x * height), int(l.y * width)
                    landmarks.append([lx, ly])

                self.draw.draw_landmarks(rgb_frame, mhl, self.mph.HAND_CONNECTIONS, landmark_drawing_spec=self.draw.
                                         DrawingSpec(color=(200, 211, 232), thickness=2, circle_radius=3))

                prediction = self.model.predict([landmarks])
                index = np.argmax(prediction)
                gesture = self.gesture_names[index]

                if time.time() - self.last_gesture_time > 1.5:
                    if not self.gesture_active:
                        inverted = {v: k for k, v in keyboard_meanings.items()}

                        if gesture in gesture_meanings.keys():
                            pyautogui.hotkey(inverted[gesture_meanings[gesture]])
                            self.gesture_active = True

                        self.last_gesture_time = time.time()
                else:
                    self.gesture_active = False

        img = Image.fromarray(rgb_frame)
        img = ImageTk.PhotoImage(img)
        self.camera_label.configure(image=img)
        self.camera_label.image = img

        self.root.after(30, self.update)

    def open_settings_window(self):
        self.root.destroy()
        settings_window = tk.Tk()
        settings = SettingsWindow(settings_window)
        settings.run()

    def open_start_window(self):
        self.root.destroy()
        start_window = tk.Tk()
        start = StartWindow(start_window)
        start.run()

    def run(self):
        self.root.mainloop()


class SettingsWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Settings")
        self.root.geometry("670x900")

        background_image = Image.open("icons/background.jpg")
        background_image = ImageTk.PhotoImage(background_image)
        self.background_label = tk.Label(root, image=background_image)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
        self.background_label.image = background_image

        self.textboxes = {}

        self.create_meanings()
        self.create_back_button()
        self.create_user_manual_button()

    def create_meanings(self):
        for i, (gesture, meaning) in enumerate(gesture_meanings.items()):
            gesture_image = Image.open(f"icons/{gesture}.png")
            gesture_image = gesture_image.resize((110, 110))
            gesture_image = ImageTk.PhotoImage(gesture_image)
            gesture_image_label = tk.Label(self.root, image=gesture_image, bg="#c9d4e8")
            gesture_image_label.image = gesture_image
            gesture_image_label.grid(row=i, column=0, padx=(5, 0), pady=1)

            meaning_label = tk.Label(self.root, text=meaning, font=("Montserrat", 10), bg="#c9d4e8")
            meaning_label.grid(row=i, column=1, sticky="w", padx=3, pady=1)

            change_button = tk.Button(self.root, text="CHANGE", font=("Montserrat", 9), fg="#ffffff", bg="#8da4d5",
                                command=lambda index=i: self.toggle_textbox(index), padx=10, pady=5)
            change_button.grid(row=i, column=2, padx=(5, 0), pady=1)

    def toggle_textbox(self, index):
        if index in self.textboxes:
            textbox = self.textboxes[index]
            textbox.grid_remove()
            del self.textboxes[index]
        else:
            textbox = tk.Text(self.root, width=18, height=2, font=("Montserrat", 9), fg="white", bg="#8da4d5")
            textbox.grid(row=index, column=4, padx=(0, 5), pady=5)
            textbox.bind("<Return>", lambda event, idx=index: self.handle_enter(event, idx))
            self.textboxes[index] = textbox

    def handle_enter(self, event, index):
        textbox = self.textboxes[index]
        user_input = textbox.get("1.0", "end-1c")
        self.change_gesture(index, user_input)
        textbox.grid_remove()
        del self.textboxes[index]

    def change_gesture(self, index, user_input):
        if user_input == "shift n":
            user_input = ("shift", "n")
        elif user_input == "shift p":
            user_input = ("shift", "p")

        gesture = list(gesture_meanings.keys())[index]
        meaning = keyboard_meanings[user_input]
        gesture_meanings[gesture] = meaning

        meaning_widget = self.root.grid_slaves(row=index, column=1)[0]
        meaning_widget.config(text=meaning)

    def create_back_button(self):
        back_button = tk.Button(self.root, text="BACK", font=("Montserrat", 10), fg="#ffffff", bg="#8da4d5",
                                command=self.back_action, padx=10, pady=5)
        back_button.grid(row=len(gesture_meanings), column=0, columnspan=2, pady=20, padx=0)

    def back_action(self):
        self.root.destroy()
        root = tk.Tk()
        start = StartWindow(root)
        start.run()

    def create_user_manual_button(self):
        self.user_manual_button = tk.Button(self.root, text="USER MANUAL", font=("Montserrat", 10), fg="#ffffff",
                                            bg="#8da4d5", command=self.user_manual_action, padx=10, pady=5)
        self.user_manual_button.grid(row=len(gesture_meanings), column=2, columnspan=2, padx=0, pady=20)

    def user_manual_action(self):
        self.root.destroy()
        root = tk.Tk()
        user_manual = UserManualWindow(root)
        user_manual.run()

    def run(self):
        self.root.mainloop()


class UserManualWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("User Manual")
        self.root.geometry("530x550")

        background_image = Image.open("icons/background.jpg")
        background_image = ImageTk.PhotoImage(background_image)
        self.background_label = tk.Label(root, image=background_image)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
        self.background_label.image = background_image

        self.create_user_manual()
        self.create_back_button()

    def create_user_manual(self):
        manual_text = "The text below represents each input's meaning and\nshould serve you as a guide if you want " \
                       "to change\nthe actions of the gestures in the Settings window"
        manual_label = tk.Label(self.root, text=manual_text, font=("Montserrat", 10), bg="#c9d4e8")
        manual_label.grid(row=0, column=0, columnspan=2, sticky="w", padx=30, pady=(10, 30))

        for i, (button, meaning) in enumerate(keyboard_meanings.items()):
            button_label = tk.Label(self.root, text=button, font=("Montserrat", 10), bg="#c9d4e8")
            button_label.grid(row=i+1, column=0, sticky="w", padx=(100, 0), pady=4)

            meaning_label = tk.Label(self.root, text=meaning, font=("Montserrat", 10), bg="#c9d4e8")
            meaning_label.grid(row=i+1, column=1, sticky="w", padx=(150, 10), pady=4)

    def create_back_button(self):
        back_button = tk.Button(self.root, text="BACK", font=("Montserrat", 10), fg="#ffffff", bg="#8da4d5",
                                command=self.back_action, padx=10, pady=5)
        back_button.grid(row=len(keyboard_meanings)+1, column=0, columnspan=2, pady=(30, 10), padx=0)

    def back_action(self):
        self.root.destroy()
        root = tk.Tk()
        settings = SettingsWindow(root)
        settings.run()

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    root = tk.Tk()
    app = StartWindow(root)
    app.run()















