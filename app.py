from flask import Flask, render_template, request, send_file, redirect
import os
import cv2
import dlib
import numpy as np

app = Flask(__name__)

# Verzeichnisse für temporäre Dateien
UPLOAD_FOLDER = 'uploads/'
OUTPUT_FOLDER = 'outputs/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Erlaubte Datei-Erweiterungen
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Dlib Modelle laden
predictor_path = "shape_predictor_68_face_landmarks.dat"  # Lade das Modell von dlib
if not os.path.exists(predictor_path):
    raise FileNotFoundError(f"{predictor_path} fehlt. Lade es von: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
face_detector = dlib.get_frontal_face_detector()
face_predictor = dlib.shape_predictor(predictor_path)

def allowed_file(filename, allowed_extensions):
    """Überprüft, ob die Datei einen erlaubten Typ hat."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def extract_face(image_path):
    """
    Extrahiert ein Gesicht aus dem Bild.
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    if len(faces) == 0:
        raise ValueError("Kein Gesicht im Bild gefunden.")

    # Wähle das erste erkannte Gesicht
    face = faces[0]
    landmarks = face_predictor(gray, face)
    points = np.array([[p.x, p.y] for p in landmarks.parts()])
    return image, points

def warp_face(src_face, src_points, dest_points, dest_shape):
    """
    Passt das Gesicht vom Quellbild an das Zielgesicht an.
    """
    matrix = cv2.getAffineTransform(
        np.float32(src_points[:3]),  # Nur die ersten drei Punkte für die Transformation
        np.float32(dest_points[:3])
    )
    warped_face = cv2.warpAffine(src_face, matrix, (dest_shape[1], dest_shape[0]))
    return warped_face

def perform_face_swap(video_path, image_path, output_path):
    """
    Führt den Gesichts-Austausch durch.
    """
    # Gesicht aus dem Bild extrahieren
    face_image, face_points = extract_face(image_path)

    # Öffne das Video
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, int(cap.get(cv2.CAP_PROP_FPS)),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray_frame)

        for face in faces:
            landmarks = face_predictor(gray_frame, face)
            points = np.array([[p.x, p.y] for p in landmarks.parts()])

            # Transformiere das Gesicht ins Video
            warped_face = warp_face(face_image, face_points, points, frame.shape)
            
            convexhull = cv2.convexHull(points)
            face_mask = np.zeros_like(gray_frame)
            cv2.fillConvexPoly(face_mask, convexhull, 255)

            center = tuple(np.mean(points, axis=0).astype(int))
            frame = cv2.seamlessClone(warped_face, frame, face_mask, center, cv2.NORMAL_CLONE)

        out.write(frame)

    cap.release()
    out.release()

@app.route('/')
def index():
    """Startseite."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Datei-Upload und Verarbeitung."""
    if 'video' not in request.files or 'image' not in request.files:
        return "Missing video or image file", 400

    video_file = request.files['video']
    image_file = request.files['image']

    if video_file.filename == '' or image_file.filename == '':
        return "No video or image file selected", 400

    if video_file and allowed_file(video_file.filename, ALLOWED_VIDEO_EXTENSIONS):
        video_filename = video_file.filename
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        video_file.save(video_path)
    else:
        return "Invalid video file type", 400

    if image_file and allowed_file(image_file.filename, ALLOWED_IMAGE_EXTENSIONS):
        image_filename = image_file.filename
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        image_file.save(image_path)
    else:
        return "Invalid image file type", 400

    # Generiere das neue Video
    output_filename = f"output_{video_filename}"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    try:
        perform_face_swap(video_path, image_path, output_path)
    except ValueError as e:
        return str(e), 400

    return redirect(f"/download/{output_filename}")

@app.route('/download/<filename>')
def download_file(filename):
    """Stellt das generierte Video zum Download bereit."""
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if not os.path.exists(output_path):
        return "File not found", 404
    return send_file(output_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
