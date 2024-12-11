from flask import Flask, render_template, request, send_file, redirect
import os

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

def allowed_file(filename, allowed_extensions):
    """Überprüft, ob die Datei einen erlaubten Typ hat."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def perform_face_swap(video_path, image_path, output_path):
    """
    Simulierte Funktion, die den Face-Swap durchführt.
    Hier würde ein tatsächliches KI-Modell aufgerufen.
    """
    # In der Realität: Das Modell aufrufen und das Video verarbeiten.
    # Hier simulieren wir einfach die Generierung.
    with open(output_path, 'w') as f:
        f.write("Simuliertes generiertes Video")
    return output_path

@app.route('/')
def index():
    """Startseite."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Datei-Upload und Verarbeitung."""
    # Überprüfen, ob Video- und Bilddateien im Request vorhanden sind
    if 'video' not in request.files or 'image' not in request.files:
        return "Missing video or image file", 400

    video_file = request.files['video']
    image_file = request.files['image']

    # Überprüfen, ob beide Dateien ausgewählt wurden
    if video_file.filename == '' or image_file.filename == '':
        return "No video or image file selected", 400

    # Video überprüfen und speichern
    if video_file and allowed_file(video_file.filename, ALLOWED_VIDEO_EXTENSIONS):
        video_filename = video_file.filename
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        video_file.save(video_path)
    else:
        return "Invalid video file type", 400

    # Bild überprüfen und speichern
    if image_file and allowed_file(image_file.filename, ALLOWED_IMAGE_EXTENSIONS):
        image_filename = image_file.filename
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        image_file.save(image_path)
    else:
        return "Invalid image file type", 400

    # Generiere das neue Video
    output_filename = f"output_{video_filename}"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    perform_face_swap(video_path, image_path, output_path)

    # Generiertes Video zum Download bereitstellen
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
