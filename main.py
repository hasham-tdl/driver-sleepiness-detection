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
        self.use_webcam = False

        # ---- MODELS ----
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

        # ---- THRESHOLDS ----
        self.EYE_THRESH = 0.24
        self.MAR_THRESH = 0.70
        self.SLEEP_TIME = 1.2
        self.YAWN_TIME = 1.0

        # ---- UNCONSCIOUSNESS ----
        self.UNCONSCIOUS_EAR = 0.18
        self.HEAD_TILT_THRESH = 15

        # ---- STATE ----
        self.state = None
        self.eyes_closed_start = None
        self.yawn_start = None
        self.alarm_played = False

        # ---- TIMERS ----
        self.last_face_time = time.time()
        self.FACE_LOST_TIMEOUT = 0.5
        self.last_alarm_time = 0
        self.ALARM_COOLDOWN = 5

        # ---- SMOOTHING ----
        self.ear_history = deque(maxlen=5)
        self.mar_history = deque(maxlen=5)

        # ---- PERCLOS ----
        self.perclos = deque(maxlen=900)
        self.PERCLOS_THRESH = 0.25

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
        webcam_btn = Button(text="Use Webcam")
        play_btn = Button(text="Play")
        stop_btn = Button(text="Stop")

        load_btn.bind(on_press=self.open_file_chooser)
        webcam_btn.bind(on_press=self.start_webcam)
        play_btn.bind(on_press=self.start)
        stop_btn.bind(on_press=self.stop)

        controls.add_widget(load_btn)
        controls.add_widget(webcam_btn)
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
                self.use_webcam = False
                popup.dismiss()

        btn.bind(on_press=select)
        popup.open()

    # ---------------- WEBCAM ----------------
    def start_webcam(self, _):
        self.stop(None)
        self.use_webcam = True

        for idx in range(5):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                self.capture = cap
                break

        if not self.capture:
            print("❌ Webcam not detected")
            return

        Clock.schedule_interval(self.update, 1 / 30)

    # ---------------- CONTROLS ----------------
    def start(self, _):
        if self.use_webcam or not self.video_path:
            return
        self.stop(None)
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
            if not self.use_webcam:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)

        now = time.time()
        faces = self.detector(gray, 1)

        if len(faces) == 0:
            if now - self.last_face_time > self.FACE_LOST_TIMEOUT:
                self.state = None
                self.eyes_closed_start = None
                self.yawn_start = None
                self.perclos.clear()
                self.alarm_played = False
            self.render(gray)
            return

        self.last_face_time = now

        # ---- DRIVER FACE (largest) ----
        faces = sorted(
            faces,
            key=lambda f: (f.right() - f.left()) * (f.bottom() - f.top()),
            reverse=True
        )
        face = faces[0]

        shape = face_utils.shape_to_np(self.predictor(gray, face))

        # ---- EAR (one-eye supported) ----
        ear_values = []
        left_eye = shape[self.lStart:self.lEnd]
        right_eye = shape[self.rStart:self.rEnd]

        if np.linalg.norm(left_eye[0] - left_eye[3]) > 1:
            ear_values.append(eye_aspect_ratio(left_eye))
        if np.linalg.norm(right_eye[0] - right_eye[3]) > 1:
            ear_values.append(eye_aspect_ratio(right_eye))

        if not ear_values:
            self.render(gray)
            return

        ear = min(ear_values)
        self.ear_history.append(ear)
        ear = sum(self.ear_history) / len(self.ear_history)

        self.perclos.append(1 if ear < self.EYE_THRESH else 0)

        if ear < self.EYE_THRESH:
            if self.eyes_closed_start is None:
                self.eyes_closed_start = now
        else:
            self.eyes_closed_start = None

        mar = mouth_aspect_ratio(shape[self.mStart:self.mEnd])
        self.mar_history.append(mar)
        mar = sum(self.mar_history) / len(self.mar_history)

        if mar > self.MAR_THRESH:
            if self.yawn_start is None:
                self.yawn_start = now
        else:
            self.yawn_start = None

        perclos_rate = sum(self.perclos) / len(self.perclos)
        yawning = self.yawn_start and (now - self.yawn_start) > self.YAWN_TIME

        # ---- HEAD TILT ----
        nose = shape[33]
        chin = shape[8]
        angle = np.degrees(np.arctan2(chin[1] - nose[1], chin[0] - nose[0]))

        unconscious = ear < self.UNCONSCIOUS_EAR and abs(angle) > self.HEAD_TILT_THRESH

        if unconscious or (self.eyes_closed_start and (now - self.eyes_closed_start) > self.SLEEP_TIME):
            self.state = "SLEEPING"
        elif yawning or perclos_rate > self.PERCLOS_THRESH:
            self.state = "DROWSY"
        else:
            self.state = "AWAKE"

        # ---- DRAW ----
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(gray, (x1, y1), (x2, y2), 255, 2)

        cv2.putText(gray, f"EAR: {ear:.2f}", (30, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        cv2.putText(gray, f"PERCLOS: {perclos_rate:.2f}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)

        if self.state == "DROWSY":
            cv2.putText(gray, "DROWSY", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 4)

        elif self.state == "SLEEPING":
            cv2.putText(gray, "⚠ SLEEPING", (30, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, 255, 5)

            if not self.alarm_played and (now - self.last_alarm_time) > self.ALARM_COOLDOWN:
                AlarmThread(alarm_path).start()
                self.alarm_played = True
                self.last_alarm_time = now

        self.render(gray)

    # ---------------- RENDER ----------------
    def render(self, gray):
        gray = cv2.flip(gray, 0)
        texture = Texture.create(size=(gray.shape[1], gray.shape[0]), colorfmt="luminance")
        texture.blit_buffer(gray.tobytes(), colorfmt="luminance", bufferfmt="ubyte")
        self.image.texture = texture


if __name__ == "__main__":
    DriverSleepinessVideoApp().run()

