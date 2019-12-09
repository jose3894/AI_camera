from flask import Flask, render_template, Response
from camera import CameraStream
import cv2

import numpy as np #importacion de numpy
from utilidades import dibujar_deteccion #importación de libreria de utilidades
from yolo import YOLO#importacion de red yolo
import json#importacion de libreria de apertura de json
import time#importacion de libreria de medicion de tiempos

ruta_configuracion = '/app/Production/configuracion.json'  # ruta a archivo de configuracion
ruta_pesos = '/app/Production/yolo_anchors_5.h5'  # ruta a red yolo ya entrenada

with open(ruta_configuracion) as buffer_configuracion:  # Cargado de fichero de configuracion
    configuracion = json.load(buffer_configuracion)

yolo = YOLO(backend=configuracion['model']['backend'],
            tamano_entrada=configuracion['model']['tamano_entrada'],
            etiquetas=configuracion['model']['etiquetas'],
            max_cajas_por_imagen=configuracion['model']['max_cajas_por_imagen'],
            tamanos_base=configuracion['model']['tamanos_base'])  # Creacion del modelo Yolo segun el fichero de configuracion

yolo.cargar_pesos(ruta_pesos)  # Cargado de pesos de red Yolo previamente entrenada

app = Flask(__name__)

cap = CameraStream().start()

num_frame = 0

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen_frame():
    global num_frame
    """Video streaming generator function."""
    while cap:
        frame = cap.read()
        if num_frame == 0:
            frame = frame[:, int(frame.shape[1] / 2) - int(frame.shape[0] / 2):int(frame.shape[1] / 2) + int(frame.shape[0] / 2), :]
            [cajas, caracteristicas] = yolo.predecir(frame, 0.25, 0.4)  # prediccion de yolo
            imageaux = dibujar_deteccion(frame, cajas, configuracion['model']['etiquetas'])  # dibujado de detecciones de yolo en la imagen

            convert = cv2.imencode('.jpg', frame)[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + convert + b'\r\n') # concate frame one by one and show result
            num_frame += 1

        if num_frame == 10:
            num_frame = 0


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)