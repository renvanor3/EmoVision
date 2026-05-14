"""
app.py
Interface graphique EmoVision.
Permet de charger une image statique depuis le disque, de détecter
le visage présent, et d'afficher l'émotion prédite par le modèle CNN.
"""

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import cv2
from PIL import Image, ImageTk

sys.path.append(str(Path(__file__).parent.parent))

from predict.face_detector import detect_faces, extract_face
from predict.predictor import load_model, predict_emotion


CANVAS_WIDTH = 640
CANVAS_HEIGHT = 480


class EmoVisionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EmoVision - Reconnaissance d'émotions")
        self.root.geometry("800x650")

        self.model = load_model()
        self.current_image_tk = None

        self._build_ui()

    def _build_ui(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(pady=10)

        tk.Button(
            top_frame,
            text="Charger une image",
            command=self.load_image,
            width=20,
        ).pack(side=tk.LEFT, padx=5)

        self.canvas = tk.Canvas(
            self.root,
            width=CANVAS_WIDTH,
            height=CANVAS_HEIGHT,
            bg="gray20",
            highlightthickness=0,
        )
        self.canvas.pack(pady=10)

        self.result_label = tk.Label(
            self.root,
            text="Aucune image chargée",
            font=("Arial", 14),
        )
        self.result_label.pack(pady=10)

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Choisir une image",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp"),
                ("Tous les fichiers", "*.*"),
            ],
        )
        if not path:
            return

        image_bgr = cv2.imread(path)
        if image_bgr is None:
            self.result_label.config(text=f"Impossible de charger : {path}")
            return

        self.process_image(image_bgr)

    def process_image(self, image_bgr):
        faces = detect_faces(image_bgr)
        display = image_bgr.copy()

        if len(faces) == 0:
            result_text = "Aucun visage détecté dans l'image"
        else:
            face = max(faces, key=lambda b: b[2] * b[3])
            x, y, w, h = face

            face_img = extract_face(image_bgr, face)
            label, probas = predict_emotion(self.model, face_img)
            confidence = probas.max() * 100

            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                display,
                f"{label} ({confidence:.0f}%)",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            result_text = f"Émotion : {label}    Confiance : {confidence:.1f}%"

        self.display_image(display)
        self.result_label.config(text=result_text)

    def display_image(self, image_bgr):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail((CANVAS_WIDTH, CANVAS_HEIGHT))

        self.current_image_tk = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(
            CANVAS_WIDTH // 2,
            CANVAS_HEIGHT // 2,
            image=self.current_image_tk,
            anchor=tk.CENTER,
        )


if __name__ == "__main__":
    root = tk.Tk()
    EmoVisionApp(root)
    root.mainloop()