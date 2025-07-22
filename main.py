# main.py

import cv2
import mediapipe as mp
import time
import json
import os
import numpy as np

# Setup folders
os.makedirs("logs", exist_ok=True)

# Logger
def log_event(event):
    log = {
        "timestamp": time.strftime("%H:%M:%S"),
        "event": event
    }
    with open("logs/events.json", "a") as f:
        f.write(json.dumps(log) + "\n")
    print(log)

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# 3D facial landmark indices for head direction estimation
LANDMARK_IDS = {
    "nose_tip": 1,
    "left_eye": 33,
    "right_eye": 263,
    "mouth": 13
}

# Start webcam
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    last_looked_away = False

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        h, w, _ = frame.shape
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            # Get required landmark coordinates
            landmarks = {}
            for name, idx in LANDMARK_IDS.items():
                lm = face_landmarks.landmark[idx]
                landmarks[name] = (int(lm.x * w), int(lm.y * h))

            # Head direction: compare nose to midpoint of eyes
            nose = np.array(landmarks["nose_tip"])
            left_eye = np.array(landmarks["left_eye"])
            right_eye = np.array(landmarks["right_eye"])
            eye_mid = (left_eye + right_eye) / 2

            dx = nose[0] - eye_mid[0]

            # Threshold for looking away (tweak if needed)
            if abs(dx) > 25:
                if not last_looked_away:
                    log_event("User looked away")
                    last_looked_away = True
            else:
                if last_looked_away:
                    log_event("User looked forward again")
                    last_looked_away = False

            # Draw face mesh
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

        cv2.imshow("Head Pose Detection", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

