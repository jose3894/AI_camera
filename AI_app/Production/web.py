from flask import Flask, render_template, Response
from camera import CameraStream
import cv2
from object_detection import ObjectDetection


app = Flask(__name__)

cap = CameraStream().start()

obj = ObjectDetection()

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen_frame():
    """Video streaming generator function."""
    while cap:
        frame = cap.read()
        frame1 = frame
        frame = obj.predict(frame)
        convert = cv2.imencode('.jpg', frame1)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + convert + b'\r\n') # concate frame one by one and show result

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)