import os
import dlib
import numpy as np
import pandas as pd
import cv2

# Paths to the dlib pre-trained models
predictor_path = "data/data_dlib/shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "data/data_dlib/dlib_face_recognition_resnet_model_v1.dat"
faces_folder_path = "data/faces_from_camera"

# Initialize dlib detectors and models
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

def compute_face_descriptor(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) == 0:
        return None
    
    # Assuming one face per cropped registration image
    face = faces[0]
    shape = shape_predictor(gray, face)
    face_descriptor = facerec.compute_face_descriptor(img, shape)
    return np.array(face_descriptor)

def main():
    if not os.path.exists(faces_folder_path):
        print("[Error] No captured faces folder found! Run the registration script first.")
        return

    person_dirs = os.listdir(faces_folder_path)
    if not person_dirs:
        print("[Error] No student folders found inside faces_from_camera.")
        return

    features_dict = {}

    for person_dir in person_dirs:
        person_path = os.path.join(faces_folder_path, person_dir)
        if not os.path.isdir(person_path):
            continue
        
        print(f"[Processing] Extracting features for: {person_dir}")
        image_files = os.listdir(person_path)
        descriptors = []

        for img_file in image_files:
            img_path = os.path.join(person_path, img_file)
            desc = compute_face_descriptor(img_path)
            if desc is not None:
                descriptors.append(desc)

        if descriptors:
            # Calculate the mean of all 128D descriptors for this person
            mean_descriptor = np.mean(descriptors, axis=0)
            features_dict[person_dir] = mean_descriptor

    # Save to CSV
    if features_dict:
        df = pd.DataFrame.from_dict(features_dict, orient='index')
        df.to_csv("features_all.csv", header=False)
        print("[Success] Features successfully extracted and saved to features_all.csv!")
    else:
        print("[Warning] No valid face descriptors could be computed.")

if __name__ == "__main__":
    main()