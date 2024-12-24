import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, send_file, redirect, render_template

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
OUTPUT_FOLDER = 'outputs/'
MODEL_FOLDER = 'models/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER

MODEL_PATH = os.path.join(MODEL_FOLDER, 'faceswap_model.h5')

# Funktion, um das Modell zu laden
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Das vortrainierte Modell fehlt. Bitte laden Sie es zuerst hoch.")
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

# Funktion, um Gesichts-Landmarken zu extrahieren
def extract_face_landmarks(image_path):
    import dlib
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if not faces:
        raise ValueError("Kein Gesicht im Bild gefunden.")

    face = faces[0]
    landmarks = predictor(gray, face)
    points = np.array([[p.x, p.y] for p in landmarks.parts()])
    return image, points

# Funktion, um den Gesichtsaustausch durchzuführen
def apply_faceswap(video_path, image_path, output_path):
    source_image, source_landmarks = extract_face_landmarks(image_path)
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, int(cap.get(cv2.CAP_PROP_FPS)),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    model = load_model()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_frame)

        if faces:
            target_face = faces[0]
            target_landmarks = predictor(gray_frame, target_face)
            target_points = np.array([[p.x, p.y] for p in target_landmarks.parts()])

            matrix, _ = cv2.findHomography(source_landmarks, target_points)
            warped_face = cv2.warpPerspective(source_image, matrix, (frame.shape[1], frame.shape[0]))

            mask = np.zeros_like(gray_frame)
            cv2.fillConvexPoly(mask, cv2.convexHull(target_points), 255)

            frame = cv2.seamlessClone(warped_face, frame, mask, tuple(np.mean(target_points, axis=0).astype(int)),
                                      cv2.NORMAL_CLONE)

        out.write(frame)

    cap.release()
    out.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files or 'image' not in request.files:
        return "Video und Bild werden benötigt.", 400

    video_file = request.files['video']
    image_file = request.files['image']

    if not video_file or not image_file:
        return "Bitte laden Sie sowohl ein Video als auch ein Bild hoch.", 400

    video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)

    video_file.save(video_path)
    image_file.save(image_path)

    output_path = os.path.join(OUTPUT_FOLDER, f"output_{video_file.filename}")
    apply_faceswap(video_path, image_path, output_path)

    return redirect(f"/download/{os.path.basename(output_path)}")

@app.route('/upload_model', methods=['POST'])
def upload_model():
    if 'model' not in request.files:
        return "Bitte laden Sie ein Modell hoch.", 400

    model_file = request.files['model']
    model_path = os.path.join(MODEL_FOLDER, model_file.filename)
    model_file.save(model_path)

    if os.path.exists(model_path):
        os.rename(model_path, MODEL_PATH)
        return "Modell erfolgreich hochgeladen.", 200
    return "Fehler beim Hochladen des Modells.", 500

@app.route('/download/<filename>')
def download(filename):
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "Datei nicht gefunden.", 404

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
