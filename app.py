import os
import cv2
import numpy as np
from flask import Flask, request, send_file, redirect, render_template
import mediapipe as mp

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
OUTPUT_FOLDER = 'outputs/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Mediapipe initialisieren
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Funktion, um Gesichts-Landmarken zu extrahieren
def extract_face_landmarks(image_path):
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Mediapipe Gesichtserkennung und Landmarkenerkennung
    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        raise ValueError("Kein Gesicht im Bild gefunden.")

    face_landmarks = results.multi_face_landmarks[0]
    landmarks = [(int(lm.x * image.shape[1]), int(lm.y * image.shape[0])) for lm in face_landmarks.landmark]

    return image, np.array(landmarks)

# Funktion, um den Gesichtsaustausch durchzuführen
def apply_faceswap(video_path, image_path, output_path):
    source_image, source_landmarks = extract_face_landmarks(image_path)
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, int(cap.get(cv2.CAP_PROP_FPS)),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            target_landmarks = results.multi_face_landmarks[0]
            target_points = np.array([
                (int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]))
                for lm in target_landmarks.landmark
            ])

            matrix, _ = cv2.findHomography(source_landmarks, target_points)
            warped_face = cv2.warpPerspective(source_image, matrix, (frame.shape[1], frame.shape[0]))

            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
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

@app.route('/download/<filename>')
def download(filename):
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "Datei nicht gefunden.", 404

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
