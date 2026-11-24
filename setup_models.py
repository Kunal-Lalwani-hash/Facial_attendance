import os
import urllib.request
import bz2

DLIB_DIR = "data/data_dlib"
FACES_DIR = "data/faces_from_camera"

MODELS = {
    "shape_predictor_68_face_landmarks.dat": "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
    "dlib_face_recognition_resnet_model_v1.dat": "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
}

def ensure_data_folders():
    os.makedirs(DLIB_DIR, exist_ok=True)
    os.makedirs(FACES_DIR, exist_ok=True)
    print(f"[OK] Created directory structure: '{DLIB_DIR}' and '{FACES_DIR}'")

    for model_name, url in MODELS.items():
        target_path = os.path.join(DLIB_DIR, model_name)
        if os.path.exists(target_path):
            print(f"[OK] {model_name} already exists.")
            continue

        bz2_path = target_path + ".bz2"
        print(f"[Downloading] {model_name} from dlib.net (this may take a moment)...")
        urllib.request.urlretrieve(url, bz2_path)

        print(f"[Extracting] Decompressing {bz2_path}...")
        with bz2.BZ2File(bz2_path, "rb") as src, open(target_path, "wb") as dst:
            dst.write(src.read())

        os.remove(bz2_path)
        print(f"[Success] Saved {target_path}!")

if __name__ == "__main__":
    ensure_data_folders()
