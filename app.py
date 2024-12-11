from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)

# Verzeichnisse für temporäre Uploads
UPLOAD_FOLDER = 'uploads/'
VIDEO_FOLDER = os.path.join(UPLOAD_FOLDER, 'videos/')
IMAGE_FOLDER = os.path.join(UPLOAD_FOLDER, 'images/')

os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Erlaubte Dateitypen
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}

app.config['VIDEO_FOLDER'] = VIDEO_FOLDER
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER

def allowed_file(filename, allowed_extensions):
    """Überprüft, ob die Datei einen erlaubten Typ hat."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/')
def index():
    """Startseite."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Verarbeitet den Upload von Video und Bild."""
    if 'video' not in request.files or 'image' not in request.files:
        return "Both video and image files are required.", 400

    video = request.files['video']
    image = request.files['image']

    # Überprüfe Video
    if video.filename == '' or not allowed_file(video.filename, ALLOWED_VIDEO_EXTENSIONS):
        return "Invalid or missing video file.", 400

    # Überprüfe Bild
    if image.filename == '' or not allowed_file(image.filename, ALLOWED_IMAGE_EXTENSIONS):
        return "Invalid or missing image file.", 400

    # Speichere Dateien
    video_path = os.path.join(app.config['VIDEO_FOLDER'], video.filename)
    image_path = os.path.join(app.config['IMAGE_FOLDER'], image.filename)

    video.save(video_path)
    image.save(image_path)

    # Bestätigung der Speicherung
    return f"Video and image uploaded successfully! Video: {video.filename}, Image: {image.filename}"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
