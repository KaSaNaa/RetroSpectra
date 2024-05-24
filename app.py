from flask import Flask, request
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from flask import request, send_from_directory
import os

app = Flask(__name__)
CORS(app)

model = load_model('model_file_30epochs.h5')
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

def process_frame(frame): ...

@app.route('/process_frame', methods=['POST'])
def process_frame_route():
    image = cv2.imdecode(np.frombuffer(request.files['image'].read(), np.uint8), cv2.IMREAD_COLOR)
    processed_image = process_frame(image)
    ret, buffer = cv2.imencode('.jpg', processed_image)
    return buffer.tobytes()

if __name__ == '__main__':
    app.run(debug=True)

