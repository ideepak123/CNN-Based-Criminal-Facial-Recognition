from flask import Flask, render_template, request
import os
import cv2
from werkzeug.utils import secure_filename
from face_detection import detect_face
from recognize_cnn import recognize_person_cnn
from recognize_face import recognize_person
from criminal_data import get_criminal_details

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():

    if 'image' not in request.files:
        return "No image uploaded"

    file = request.files['image']

    if file.filename == '':
        return "No selected image"

    filename = secure_filename(file.filename)
    filepath = os.path.join(
        app.config['UPLOAD_FOLDER'],
        filename
    )

    file.save(filepath)

    # Detect face
    processed_image = detect_face(filepath)

    if processed_image is not None:
        cv2.imwrite(filepath, processed_image)

   # CNN Recognition
    criminal_name, confidence = (
    recognize_person_cnn(filepath)
)
    criminal_info = get_criminal_details(
        criminal_name
    )
    return render_template(
    'result.html',
    image=filename,
    criminal_name=criminal_name,
    confidence=confidence,
    criminal_info=criminal_info
)


if __name__ == '__main__':
    app.run(debug=True)