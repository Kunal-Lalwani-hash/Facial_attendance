import cv2
import dlib
import os

detector = dlib.get_frontal_face_detector()

def start_capture():
    print("\n--- Student Biometric Face Registration ---")
    
    roll_no = input("Enter Student Roll Number (Required, e.g., 23T7030): ").strip().upper()
    student_name = input("Enter Student Full Name (Required, e.g., Kunal Lalwani): ").strip()
    
    if not roll_no or not student_name:
        print("[ERROR] Both Roll Number and Student Name are strictly required! Exiting.")
        return

    identifier = f"{roll_no}_{student_name}"
    path = f"data/faces_from_camera/person_{identifier}"
    os.makedirs(path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0
    
    print(f"\n[INFO] Starting camera for {roll_no} - {student_name}...")
    print("[INFO] Click the webcam window, press 's' to save 5-10 times, then 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Could not read from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow(f"Capturing {roll_no} - Press 's' to Save, 'q' to Quit", frame)
        
        key = cv2.waitKey(1000) & 0xFF
        if key == ord('s'):
            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cropped_face = frame[max(0, y):y+h, max(0, x):x+w]
                if cropped_face.size != 0:
                    count += 1
                    file_path = f"{path}/img_{count}.jpg"
                    cv2.imwrite(file_path, cropped_face)
                    print(f"[Saved] {file_path}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[SUCCESS] Captured {count} images for {roll_no} ({student_name})! Run features_extraction_to_csv.py next.")

if __name__ == "__main__":
    start_capture()
