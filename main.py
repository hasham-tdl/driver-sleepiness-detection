from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserIconView
from kivy.clock import Clock
from kivy.graphics.texture import Texture

import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist
from threading import Thread
import playsound
import time
from collections import deque

# ---------------- PATHS ----------------
predictor_path = r"shape_predictor\shape_predictor_68_face_landmarks.dat"
video_folder = r"videos"
alarm_path = r"audio\audio_file1.wav"

# ---------------- ALARM THREAD ----------------
class AlarmThread(Thread):
    def __init__(self, path):
        super().__init__(daemon=True)
        self.path = path

    def run(self):
        playsound.playsound(self.path)

# ---------------- METRICS ----------------
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    D = dist.euclidean(mouth[12], mouth[16])
    return (A + B + C) / (3.0 * D)

# ---------------- APP ----------------
class DriverSleepinessVideoApp(App):

    def build(self):
        self.capture = None
        self.video_path = None

        # ---- MODELS ----
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

        # ---- THRESHOLDS ----
        self.EYE_THRESH = 0.21
        self.MAR_THRESH = 0.75
        self.SLEEP_TIME = 2.0
        self.YAWN_TIME = 1.5

        # ---- STATE ----
        self.state = None
        self.eyes_closed_start = None
        self.yawn_start = None
        self.alarm_played = False

        # ---- PERCLOS ----
        self.perclos = deque(maxlen=1800)

        # ---- LANDMARK IDS ----
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (self.mStart, self.mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

        # ---- UI ----
        layout = BoxLayout(orientation="vertical")
        self.image = Image()
        layout.add_widget(self.image)

        controls = BoxLayout(size_hint_y=None, height=50)
        load_btn = Button(text="Load Video")
        play_btn = Button(text="Play")
        stop_btn = Button(text="Stop")

        load_btn.bind(on_press=self.open_file_chooser)
        play_btn.bind(on_press=self.start)
        stop_btn.bind(on_press=self.stop)

        controls.add_widget(load_btn)
        controls.add_widget(play_btn)
        controls.add_widget(stop_btn)
        layout.add_widget(controls)

        return layout

    # ---------------- FILE CHOOSER ----------------
    def open_file_chooser(self, _):
        chooser = FileChooserIconView(path=video_folder, filters=["*.mp4"])
        box = BoxLayout(orientation="vertical")
        box.add_widget(chooser)

        btn = Button(text="Select", size_hint_y=None, height=50)
        box.add_widget(btn)

        popup = Popup(title="Select Video", content=box, size_hint=(0.9, 0.9))

        def select(_):
            if chooser.selection:
                self.video_path = chooser.selection[0]
                popup.dismiss()

        btn.bind(on_press=select)
        popup.open()

    # ---------------- CONTROLS ----------------
    def start(self, _):
        if not self.video_path:
            return
        self.capture = cv2.VideoCapture(self.video_path)
        Clock.schedule_interval(self.update, 1 / 30)

    def stop(self, _):
        if self.capture:
            Clock.unschedule(self.update)
            self.capture.release()
            self.capture = None
            self.image.texture = None

    # ---------------- MAIN LOOP ----------------
    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)

        faces = self.detector(gray, 0)
        now = time.time()

        # ---- NO FACE → NO STATE ----
        if len(faces) == 0:
            self.state = None
            self.eyes_closed_start = None
            self.yawn_start = None
            self.perclos.clear()
            self.alarm_played = False
        else:
            for face in faces:
                shape = face_utils.shape_to_np(self.predictor(gray, face))

                ear = (
                    eye_aspect_ratio(shape[self.lStart:self.lEnd]) +
                    eye_aspect_ratio(shape[self.rStart:self.rEnd])
                ) / 2.0

                self.perclos.append(1 if ear < self.EYE_THRESH else 0)

                if ear < self.EYE_THRESH:
                    if self.eyes_closed_start is None:
                        self.eyes_closed_start = now
                else:
                    self.eyes_closed_start = None

                mar = mouth_aspect_ratio(shape[self.mStart:self.mEnd])

                if mar > self.MAR_THRESH:
                    if self.yawn_start is None:
                        self.yawn_start = now
                else:
                    self.yawn_start = None

                perclos_rate = sum(self.perclos) / max(len(self.perclos), 1)
                yawning = self.yawn_start and (now - self.yawn_start) > self.YAWN_TIME

                if self.eyes_closed_start and (now - self.eyes_closed_start) > self.SLEEP_TIME:
                    self.state = "SLEEPING"
                elif yawning or perclos_rate > 0.4:
                    self.state = "DROWSY"
                else:
                    self.state = "AWAKE"

                x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                cv2.rectangle(gray, (x1, y1), (x2, y2), 255, 2)
                break

        # ---- DISPLAY STATE ONLY IF FACE EXISTS ----
        if self.state == "DROWSY":
            cv2.putText(gray, "DROWSY", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 4)

        elif self.state == "SLEEPING":
            cv2.putText(gray, "⚠ SLEEPING", (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, 255, 5)
            if not self.alarm_played:
                AlarmThread(alarm_path).start()
                self.alarm_played = True

        gray = cv2.flip(gray, 0)
        texture = Texture.create(
            size=(gray.shape[1], gray.shape[0]),
            colorfmt="luminance"
        )
        texture.blit_buffer(gray.tobytes(), colorfmt="luminance", bufferfmt="ubyte")
        self.image.texture = texture


if __name__ == "__main__":
    DriverSleepinessVideoApp().run()
