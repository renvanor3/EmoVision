import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog

import cv2
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

sys.path.append(str(Path(__file__).parent.parent))

from predict.face_detector import detect_faces, extract_face
from predict.predictor import CLASS_NAMES, load_model, predict_emotion


PROJECT_DIR = Path(__file__).parent.parent
SNAPSHOTS_DIR = PROJECT_DIR / "snapshots"
SNAPSHOTS_DIR.mkdir(exist_ok=True)

CANVAS_WIDTH = 640
CANVAS_HEIGHT = 480
WEBCAM_DELAY_MS = 30


class EmoVisionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EmoVision - Reconnaissance d'émotions")
        self.root.geometry("1150x700")

        self.model = load_model()
        self.current_image_tk = None
        self.last_annotated_frame = None

        self.webcam_active = False
        self.capture = None

        self._build_ui()
        self._reset_histogram()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(pady=10)

        tk.Button(
            top_frame,
            text="Charger une image",
            command=self.load_image,
            width=20,
        ).pack(side=tk.LEFT, padx=5)

        self.webcam_button = tk.Button(
            top_frame,
            text="Démarrer la webcam",
            command=self.toggle_webcam,
            width=20,
        )
        self.webcam_button.pack(side=tk.LEFT, padx=5)

        tk.Button(
            top_frame,
            text="Capturer un instantané",
            command=self.save_snapshot,
            width=20,
        ).pack(side=tk.LEFT, padx=5)

        middle_frame = tk.Frame(self.root)
        middle_frame.pack(pady=10)

        self.canvas = tk.Canvas(
            middle_frame,
            width=CANVAS_WIDTH,
            height=CANVAS_HEIGHT,
            bg="gray20",
            highlightthickness=0,
        )
        self.canvas.pack(side=tk.LEFT, padx=10)

        self.fig = Figure(figsize=(4.5, 4.8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.fig_canvas = FigureCanvasTkAgg(self.fig, master=middle_frame)
        self.fig_canvas.get_tk_widget().pack(side=tk.LEFT, padx=10)

        self.result_label = tk.Label(
            self.root,
            text="Aucune image chargée",
            font=("Arial", 14),
        )
        self.result_label.pack(pady=10)

    def _draw_histogram(self, values, predicted_idx=None):
        self.ax.clear()
        if predicted_idx is None:
            colors = ["lightgray"] * len(CLASS_NAMES)
        else:
            colors = ["steelblue"] * len(CLASS_NAMES)
            colors[predicted_idx] = "green"
        self.ax.bar(CLASS_NAMES, values, color=colors)
        self.ax.set_ylim(0, 1)
        self.ax.set_ylabel("Probabilité")
        self.ax.set_title("Distribution des émotions")
        self.ax.tick_params(axis="x", rotation=45)
        self.fig.tight_layout()
        self.fig_canvas.draw()

    def _reset_histogram(self):
        self._draw_histogram([0] * len(CLASS_NAMES))

    def load_image(self):
        if self.webcam_active:
            self._stop_webcam()

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

    def toggle_webcam(self):
        if self.webcam_active:
            self._stop_webcam()
        else:
            self._start_webcam()

    def _start_webcam(self):
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            self.result_label.config(text="Erreur : webcam non accessible")
            self.capture = None
            return
        self.webcam_active = True
        self.webcam_button.config(text="Arrêter la webcam")
        self._webcam_loop()

    def _stop_webcam(self):
        self.webcam_active = False
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.webcam_button.config(text="Démarrer la webcam")

    def _webcam_loop(self):
        if not self.webcam_active or self.capture is None:
            return

        ret, frame = self.capture.read()
        if ret:
            self.process_image(frame)

        self.root.after(WEBCAM_DELAY_MS, self._webcam_loop)

    def process_image(self, image_bgr):
        faces = detect_faces(image_bgr)
        display = image_bgr.copy()

        if len(faces) == 0:
            result_text = "Aucun visage détecté"
            self._reset_histogram()
        else:
            face = max(faces, key=lambda b: b[2] * b[3])
            x, y, w, h = face

            face_img = extract_face(image_bgr, face)
            label, probas = predict_emotion(self.model, face_img)
            confidence = probas.max() * 100
            predicted_idx = int(probas.argmax())

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
            self._draw_histogram(probas, predicted_idx)

        self.last_annotated_frame = display
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

    def save_snapshot(self):
        if self.last_annotated_frame is None:
            self.result_label.config(text="Aucune image à sauvegarder")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = SNAPSHOTS_DIR / f"snapshot_{timestamp}.png"
        cv2.imwrite(str(filename), self.last_annotated_frame)
        self.result_label.config(text=f"Instantané sauvegardé : {filename.name}")


    def _on_close(self):
        if self.capture is not None:
            self.capture.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    EmoVisionApp(root)
    root.mainloop()