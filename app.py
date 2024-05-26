from flask import Flask, request
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

app = Flask(__name__)
CORS(app)

# load the model and the face cascade
model = load_model('model_file_30epochs.h5')
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # detect faces in the frame
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    
    # iterate over the faces and predict the emotion
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        resized = cv2.resize(roi_gray, (48, 48))
        normalize = resized / 255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return frame

@app.route('/process_frame', methods=['POST'])
def process_frame_route():
    frame = cv2.imdecode(np.frombuffer(request.files['frame'].read(), np.uint8), cv2.IMREAD_COLOR)
    processed_frame = process_frame(frame)
    ret, buffer = cv2.imencode('.jpg', processed_frame)
    return buffer.tobytes()

if __name__ == '__main__':
    app.run(debug=True)
