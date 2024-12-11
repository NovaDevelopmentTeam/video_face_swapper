from flask import Flask, render_template, request, redirect
import os

app = Flask(__name__)

# Verzeichnisse für temporäre Videos und Bilder
UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename, allowed_extensions):
    """Überprüft, ob die Datei einen erlaubten Typ hat."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/')
def index():
    """Startseite."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Datei-Upload-Logik für Video und Bild."""
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

    return f"Video {video_filename} and image {image_filename} uploaded successfully!"

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
