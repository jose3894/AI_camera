from flask import Flask, render_template, Response
from camera import CameraStream
import cv2
from darkflow.net.build import TFNet
import numpy as np


app = Flask(__name__)

cap = CameraStream().start()


options = {"model": "/app/Production/tiny-yolo-voc.cfg",
           "load": "/app/Production/tiny-yolo-voc.weights",
           "threshold": 0.1}

tfnet = TFNet(options)


def boxing(original_img, predictions):
    newImage = np.copy(original_img)

    for result in predictions:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        confidence = result['confidence']
        label = result['label'] + " " + str(round(confidence, 3))

        if confidence > 0.3:
            newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255, 0, 0), 3)
            newImage = cv2.putText(newImage, label, (top_x, top_y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8,
                                   (0, 230, 0), 1, cv2.LINE_AA)

    return newImage


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen_frame():
    """Video streaming generator function."""
    while cap:
        frame = cap.read()
        frame = np.asarray(frame)
        results = tfnet.return_predict(frame)
        new_frame = boxing(frame, results)
        convert = cv2.imencode('.jpg', new_frame)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + convert + b'\r\n') # concate frame one by one and show result


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)