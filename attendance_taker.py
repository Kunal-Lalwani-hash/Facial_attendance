import cv2
import dlib
import numpy as np
import pandas as pd
import sqlite3
import os
import math
import winsound
from datetime import datetime
from scipy.spatial.distance import euclidean


def calculate_ear(eye):
    """Eye Aspect Ratio for blink detection."""
    return (euclidean(eye[1], eye[5]) + euclidean(eye[2], eye[4])) / (2.0 * euclidean(eye[0], eye[3]))

predictor_path = "data/data_dlib/shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "data/data_dlib/dlib_face_recognition_resnet_model_v1.dat"
csv_path = "features_all.csv"
DB_NAME = "attendance.db"

SUBJECTS = [
    "System-on-Chip Design",
    "Antenna Wave Propagation",
    "Operating Systems",
    "Data Structures & Algorithms",
    "Major Project Lab"
]

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance_v2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_no TEXT NOT NULL,
            name TEXT NOT NULL,
            subject TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def log_attendance_to_db(roll_no, name, subject):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")
    
    cursor.execute("SELECT * FROM attendance_v2 WHERE roll_no = ? AND subject = ? AND date = ?", (roll_no, subject, today))
    if not cursor.fetchone():
        cursor.execute("INSERT INTO attendance_v2 (roll_no, name, subject, date, time) VALUES (?, ?, ?, ?, ?)",
                       (roll_no, name, subject, today, current_time))
        conn.commit()
        print(f"[Attendance Logged] {roll_no} | {name} ({subject}) at {current_time}")
    conn.close()

def parse_label(raw_label):
    clean = str(raw_label).replace("person_", "")
    if "_" in clean:
        parts = clean.split("_", 1)
        return parts[0], parts[1]
    return clean, clean

def load_known_faces():
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return [], []
    df = pd.read_csv(csv_path, header=None)
    return df.iloc[:, 0].values, df.iloc[:, 1:].values

def main():
    init_db()
    print("\n--- Select Active Subject for Attendance Session ---")
    for idx, sub in enumerate(SUBJECTS, 1):
        print(f"{idx}. {sub}")
    choice = input("Enter subject number (default 1): ").strip()
    active_subject = SUBJECTS[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(SUBJECTS) else SUBJECTS[0]
    print(f"[Info] Active Subject set to: {active_subject}")

    names, known_descriptors = load_known_faces()
    if len(names) == 0:
        print("[Error] No registered faces found in CSV.")
        return

    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(predictor_path)
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)

    cap = cv2.VideoCapture(0)
    frame_skip_rate = 10
    frame_count = 0
    tracked_faces = []

    print("[Info] Starting optimized webcam. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_count % frame_skip_rate == 0:
            # --- Heavy AI Frame: detect, recognize, start trackers ---
            faces = detector(gray)
            tracked_faces = []

            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                label = "Unknown"
                color = (0, 0, 255)

                shape = shape_predictor(gray, face)
                face_descriptor = np.array(facerec.compute_face_descriptor(frame, shape))

                # --- EAR Blink Detection ---
                left_eye = np.array([(shape.part(i).x, shape.part(i).y) for i in range(36, 42)])
                right_eye = np.array([(shape.part(i).x, shape.part(i).y) for i in range(42, 48)])
                avg_ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0

                distances = np.linalg.norm(known_descriptors - face_descriptor, axis=1)
                min_index = np.argmin(distances)

                if distances[min_index] < 0.45:
                    roll_no, student_name = parse_label(names[min_index])
                    label = f"{roll_no} | {student_name}"
                    color = (0, 255, 255)  # Yellow: Recognized, waiting for blink

                    if avg_ear < 0.21:
                        log_attendance_to_db(roll_no, student_name, active_subject)
                        winsound.Beep(2500, 200)
                        color = (0, 255, 0)  # Green: Blink verified

                # Start a dlib correlation tracker for this face
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(x, y, x + w, y + h)
                tracker.start_track(frame, rect)
                tracked_faces.append({'tracker': tracker, 'label': label, 'color': color})

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        else:
            # --- Intermediate Frame: update trackers, reuse saved label/color ---
            for tf in tracked_faces:
                tf['tracker'].update(frame)
                pos = tf['tracker'].get_position()
                x, y = int(pos.left()), int(pos.top())
                w, h = int(pos.width()), int(pos.height())
                label = tf['label']
                color = tf['color']

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        cv2.putText(frame, f"Subject: {active_subject}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 215, 0), 2)
        frame_count += 1

        cv2.imshow("Live Subject-Wise Attendance Taker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
