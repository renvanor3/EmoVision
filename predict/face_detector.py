"""
face_detector.py
Module de détection de visage utilisant le classifieur en cascade de
Haar fourni par OpenCV (haarcascade_frontalface_default).
Fournit detect_faces pour obtenir les bounding boxes des visages
et extract_face pour récupérer la région d'un visage donné.
"""

import sys
from pathlib import Path

import cv2


PROJECT_DIR = Path(__file__).parent.parent

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_cascade = cv2.CascadeClassifier(CASCADE_PATH)


def detect_faces(image, scale_factor=1.2, min_neighbors=5, min_size=(40, 40)):
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    faces = _cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )
    return faces


def extract_face(image, bbox):
    x, y, w, h = bbox
    return image[y:y + h, x:x + w]


def draw_faces(image, faces, color=(0, 255, 0), thickness=2):
    output = image.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)
    return output


def capture_from_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la webcam")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Impossible de capturer une image depuis la webcam")
    return frame


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        image = cv2.imread(image_path)
        if image is None:
            print(f"Impossible de charger l'image : {image_path}")
            sys.exit(1)
        print(f"Image chargée : {image_path}")
    else:
        print("Capture depuis la webcam, regarde l'objectif...")
        image = capture_from_webcam()

    faces = detect_faces(image)
    print(f"Visages détectés : {len(faces)}")
    for i, (x, y, w, h) in enumerate(faces):
        print(f"  Visage {i + 1} : position=({x},{y}), taille={w}x{h}")

    output = draw_faces(image, faces)
    cv2.imshow("Detection de visage - appuie sur une touche pour quitter", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()