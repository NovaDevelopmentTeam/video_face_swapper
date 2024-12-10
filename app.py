from flask import Flask, render_template, request, redirect, url_for
import os
import requests
from requests.auth import HTTPBasicAuth
import base64

app = Flask(__name__)

# Verzeichnis für temporäre Videos auf der Render-Seite
UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Erlaubte Video-Dateitypen
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# GitHub API Details
GITHUB_API_URL = "https://api.github.com/repos/DEIN_USERNAME/DEIN_REPOSITORY/contents/"
ACCESS_TOKEN = 'DEIN_PERSONAL_ACCESS_TOKEN'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    # Anzeige der aktuellen Videos auf Render
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Video an GitHub senden
        upload_video_to_github(filename, file_path)

        return redirect(url_for('index'))
    else:
        return "File type not allowed", 400

def upload_video_to_github(filename, video_file_path):
    """
    Lädt ein Video von der Render-Seite auf GitHub hoch.
    """
    with open(video_file_path, 'rb') as file:
        video_data = file.read()

    # Base64 Kodierung des Videos
    video_base64 = base64.b64encode(video_data).decode('utf-8')

    # GitHub API Request zum Hochladen
    file_path = 'videos/' + filename  # Zielordner auf GitHub
    message = f"Add {filename} video"

    response = requests.put(
        GITHUB_API_URL + file_path,
        auth=HTTPBasicAuth('DEIN_USERNAME', ACCESS_TOKEN),
        json={
            "message": message,
            "content": video_base64
        }
    )

    if response.status_code == 201:
        print(f"Video {filename} erfolgreich auf GitHub hochgeladen.")
    else:
        print(f"Fehler beim Hochladen des Videos: {response.status_code}, {response.text}")

if __name__ == '__main__':
    app.run(debug=True)
