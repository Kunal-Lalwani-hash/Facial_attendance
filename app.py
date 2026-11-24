from flask import Flask, render_template, request, Response, jsonify, redirect, url_for
import cv2
import dlib
import numpy as np
import pandas as pd
import sqlite3
import os
import math
import io
import csv
from datetime import datetime
from scipy.spatial.distance import euclidean

try:
    import winsound
    def trigger_beep():
        winsound.Beep(2500, 200)
except Exception:
    def trigger_beep():
        pass


def calculate_ear(eye):
    # Calculate the vertical distances between eyelids
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    # Calculate the horizontal distance between the eye corners
    C = euclidean(eye[0], eye[3])
    # Calculate the Eye Aspect Ratio
    ear = (A + B) / (2.0 * C)
    return ear


app = Flask(__name__)

# --- Configuration & Paths ---
ENABLE_ANTI_SPOOFING = True  # True: Kiosk Mode (Blink Check ON), False: Crowd Mode (Multi-Face Fast, Blink OFF)

os.makedirs("data/data_dlib", exist_ok=True)
os.makedirs("data/faces_from_camera", exist_ok=True)
predictor_path = "data/data_dlib/shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "data/data_dlib/dlib_face_recognition_resnet_model_v1.dat"
csv_path = "features_all.csv"
DB_NAME = "attendance.db"

# Default Subjects List & Active Subject State
SUBJECTS = [
    "System-on-Chip Design",
    "Antenna Wave Propagation",
    "Operating Systems",
    "Data Structures & Algorithms",
    "Major Project Lab"
]
current_subject = SUBJECTS[0]

# --- Database Setup & Auto-Migration ---
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
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            roll_no TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            registered_on TEXT NOT NULL
        )
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_att_roll_subj_date
        ON attendance_v2 (roll_no, subject, date)
    ''')

    conn.commit()
    conn.close()

init_db()

def log_attendance_to_db(roll_no, name, subject):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")
    
    cursor.execute('''
        SELECT 1 FROM attendance_v2 
        WHERE roll_no = ? AND subject = ? AND date = ?
        LIMIT 1
    ''', (roll_no, subject, today))
    
    if not cursor.fetchone():
        cursor.execute('''
            INSERT INTO attendance_v2 (roll_no, name, subject, date, time) 
            VALUES (?, ?, ?, ?, ?)
        ''', (roll_no, name, subject, today, current_time))
        conn.commit()
        print(f"[Attendance Logged] {roll_no} - {name} ({subject}) at {current_time}")
    conn.close()

def get_attendance_records(date_filter=None, subject_filter=None):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    query = "SELECT id, roll_no, name, subject, date, time FROM attendance_v2 WHERE 1=1"
    params = []
    
    if date_filter:
        query += " AND date = ?"
        params.append(date_filter)
    if subject_filter and subject_filter != "All":
        query += " AND subject = ?"
        params.append(subject_filter)
        
    query += " ORDER BY id DESC"
    cursor.execute(query, params)
    records = cursor.fetchall()
    conn.close()
    return records

def get_registered_students():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT roll_no, name, registered_on FROM students ORDER BY roll_no ASC")
    students = cursor.fetchall()
    conn.close()
    return students

def get_today_present_count():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    cursor.execute("SELECT COUNT(DISTINCT roll_no) FROM attendance_v2 WHERE date = ?", (today,))
    count = cursor.fetchone()[0]
    conn.close()
    return count

def load_known_faces():
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return [], []
    df = pd.read_csv(csv_path, header=None)
    return df.iloc[:, 0].values, df.iloc[:, 1:].values

names, known_descriptors = load_known_faces()

# --- Load AI Models ---
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

def parse_label(raw_label):
    """Parses 'person_23T7030_Kunal Lalwani' into ('23T7030', 'Kunal Lalwani')"""
    clean = str(raw_label).replace("person_", "")
    if "_" in clean:
        parts = clean.split("_", 1)
        return parts[0], parts[1]
    return clean, clean

# --- Live Scanner Generator ---
def gen_frames():
    global current_subject, ENABLE_ANTI_SPOOFING
    cap = cv2.VideoCapture(0)
    frame_skip_rate = 4
    frame_count = 0
    tracked_faces = []
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_count % frame_skip_rate == 0:
            # --- Heavy AI Frame: detect, recognize, start trackers ---
            faces = detector(gray)
            tracked_faces = []

            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                display_label, color = "Unknown", (0, 0, 255)

                shape = shape_predictor(gray, face)
                face_descriptor = np.array(facerec.compute_face_descriptor(frame, shape))

                # --- Calculate EAR for Blink Detection ---
                left_eye = np.array([(shape.part(i).x, shape.part(i).y) for i in range(36, 42)])
                right_eye = np.array([(shape.part(i).x, shape.part(i).y) for i in range(42, 48)])
                avg_ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0

                if len(known_descriptors) > 0:
                    distances = np.linalg.norm(known_descriptors - face_descriptor, axis=1)
                    min_index = np.argmin(distances)
                    if distances[min_index] < 0.45:
                        raw_label = names[min_index]
                        roll_no, student_name = parse_label(raw_label)
                        display_label = f"{roll_no} | {student_name}"

                        if ENABLE_ANTI_SPOOFING:
                            # --- Kiosk Mode (Blink Anti-Spoofing ON) ---
                            color = (0, 255, 255)  # Yellow: Recognized, waiting for blink
                            if avg_ear < 0.21:
                                log_attendance_to_db(roll_no, student_name, current_subject)
                                trigger_beep()
                                color = (0, 255, 0)  # Green: Blink verified
                        else:
                            # --- Crowd Mode (Multi-Face Fast, Blink OFF) ---
                            color = (0, 255, 0)  # Green: Instantly marked
                            log_attendance_to_db(roll_no, student_name, current_subject)
                            trigger_beep()

                # Start a dlib correlation tracker for this face
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(x, y, x + w, y + h)
                tracker.start_track(frame, rect)
                tracked_faces.append({'tracker': tracker, 'label': display_label, 'color': color})

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.rectangle(frame, (x, y - 28), (x + w, y), color, cv2.FILLED)
                cv2.putText(frame, display_label, (x + 6, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        else:
            # --- Intermediate Frame: update trackers, reuse saved label/color ---
            for tf in tracked_faces:
                tf['tracker'].update(frame)
                pos = tf['tracker'].get_position()
                x, y = int(pos.left()), int(pos.top())
                w, h = int(pos.width()), int(pos.height())
                display_label = tf['label']
                color = tf['color']

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.rectangle(frame, (x, y - 28), (x + w, y), color, cv2.FILLED)
                cv2.putText(frame, display_label, (x + 6, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        mode_str = "Kiosk (Blink ON)" if ENABLE_ANTI_SPOOFING else "Crowd (Fast)"
        cv2.putText(frame, f"Subject: {current_subject} | Mode: {mode_str}", (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 215, 0), 2)

        frame_count += 1
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

# --- Admin Registration & Feature Extraction Logic ---
def extract_and_save_features(identifier_key, img_folder):
    global names, known_descriptors
    descriptors = []
    for img_file in os.listdir(img_folder):
        img_path = os.path.join(img_folder, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if len(faces) > 0:
            shape = shape_predictor(gray, faces[0])
            desc = facerec.compute_face_descriptor(img, shape)
            descriptors.append(np.array(desc))
            
    if descriptors:
        mean_descriptor = np.mean(descriptors, axis=0)
        label_key = f"person_{identifier_key}"
        
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            existing_df = pd.read_csv(csv_path, header=None, index_col=0)
            if label_key in existing_df.index:
                existing_df = existing_df.drop(label_key)
            new_row = pd.DataFrame([mean_descriptor], index=[label_key])
            updated_df = pd.concat([existing_df, new_row])
            updated_df.to_csv(csv_path, header=False)
        else:
            df = pd.DataFrame.from_dict({label_key: mean_descriptor}, orient='index')
            df.to_csv(csv_path, mode='w', header=False)
            
        names, known_descriptors = load_known_faces()
        return True
    return False


@app.route("/", methods=["GET", "POST"])
def index():
    global current_subject
    selected_date = request.args.get("date", "")
    selected_subject = request.args.get("subject", "All")
    
    records = get_attendance_records(selected_date if selected_date else None, selected_subject)
    total_students = len(get_registered_students())
    today_present = get_today_present_count()

    return render_template(
        "index.html",
        records=records,
        selected_date=selected_date,
        selected_subject=selected_subject,
        subjects=SUBJECTS,
        current_subject=current_subject,
        total_students=total_students,
        today_present=today_present,
        anti_spoofing=ENABLE_ANTI_SPOOFING
    )

@app.route("/toggle_mode", methods=["POST"])
def toggle_mode():
    global ENABLE_ANTI_SPOOFING
    ENABLE_ANTI_SPOOFING = not ENABLE_ANTI_SPOOFING
    return redirect(url_for("index"))

@app.route("/set_subject", methods=["POST"])
def set_subject():
    global current_subject
    new_subject = request.form.get("active_subject")
    if new_subject in SUBJECTS:
        current_subject = new_subject
    return redirect(url_for("index"))

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/attendance')
def api_attendance():
    return jsonify(get_attendance_records())

@app.route('/export_csv')
def export_csv():
    date_filter = request.args.get("date")
    subject_filter = request.args.get("subject", "All")
    records = get_attendance_records(date_filter if date_filter else None, subject_filter)
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Roll Number", "Student Name", "Subject", "Date", "Time"])
    for row in records:
        writer.writerow(row)
        
    output.seek(0)
    filename = f"attendance_{date_filter or 'all'}_{subject_filter}.csv"
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename={filename}"}
    )

# --- ADMIN PANEL ROUTES ---
@app.route('/admin', methods=["GET", "POST"])
def admin():
    students = get_registered_students()
    total_logs = len(get_attendance_records())
    
    if request.method == "POST":
        roll_no = request.form.get("roll_no", "").strip().upper()
        student_name = request.form.get("student_name", "").strip()
        
        if not roll_no or not student_name:
            return render_template("admin.html", students=students, total_logs=total_logs, subjects=SUBJECTS,
                                   message="Both Roll Number and Student Name are strictly required!", msg_type="error")

        identifier_key = f"{roll_no}_{student_name}"
        path = f"data/faces_from_camera/person_{identifier_key}"
        os.makedirs(path, exist_ok=True)

        try:
            cap = cv2.VideoCapture(0)
            count = 0
            while count < 10:
                ret, frame = cap.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)
                for face in faces:
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    cropped = frame[max(0, y):y+h, max(0, x):x+w]
                    if cropped.size != 0:
                        cv2.imwrite(f"{path}/img_{count}.jpg", cropped)
                        count += 1
                        
                cv2.imshow(f"Registering {roll_no} - {student_name}", frame)
                cv2.waitKey(200)
                
            cap.release()
            cv2.destroyAllWindows()
            
            success = extract_and_save_features(identifier_key, path)
            
            if success:
                conn = sqlite3.connect(DB_NAME)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO students (roll_no, name, registered_on)
                    VALUES (?, ?, ?)
                ''', (roll_no, student_name, datetime.now().strftime("%Y-%m-%d %H:%M")))
                conn.commit()
                conn.close()
                
                students = get_registered_students()
                return render_template("admin.html", students=students, total_logs=total_logs, subjects=SUBJECTS,
                                       message=f"Successfully registered {roll_no} ({student_name})!", msg_type="success")
            else:
                return render_template("admin.html", students=students, total_logs=total_logs, subjects=SUBJECTS,
                                       message="Could not extract features. Ensure face is clearly visible.", msg_type="error")
                
        except Exception as e:
            return render_template("admin.html", students=students, total_logs=total_logs, subjects=SUBJECTS,
                                   message=f"Camera Error: {str(e)}", msg_type="error")

    return render_template("admin.html", students=students, total_logs=total_logs, subjects=SUBJECTS)

@app.route('/admin/manual_log', methods=["POST"])
def manual_log():
    roll_no = request.form.get("roll_no", "").strip().upper()
    student_name = request.form.get("student_name", "").strip()
    subject = request.form.get("subject", "").strip()
    
    if roll_no and student_name and subject:
        log_attendance_to_db(roll_no, student_name, subject)
    return redirect(url_for("admin"))

@app.route('/admin/delete_log/<int:log_id>', methods=["POST"])
def delete_log(log_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM attendance_v2 WHERE id = ?", (log_id,))
    conn.commit()
    conn.close()
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
