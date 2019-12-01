import os
import cv2 # importacion de opencv
import numpy as np #importacion de numpy
from utilidades import dibujar_deteccion #importación de libreria de utilidades
from yolo import YOLO#importacion de red yolo
import json#importacion de libreria de apertura de json
import time#importacion de libreria de medicion de tiempos

os.system('xhost +')

ruta_configuracion = '/app/Testing/configuracion.json'  # ruta a archivo de configuracion
ruta_pesos = '/app/Testing/yolo_anchors_5.h5'  # ruta a red yolo ya entrenada

with open(ruta_configuracion) as buffer_configuracion:  # Cargado de fichero de configuracion
    configuracion = json.load(buffer_configuracion)

yolo = YOLO(backend=configuracion['model']['backend'],
            tamano_entrada=configuracion['model']['tamano_entrada'],
            etiquetas=configuracion['model']['etiquetas'],
            max_cajas_por_imagen=configuracion['model']['max_cajas_por_imagen'],
            tamanos_base=configuracion['model']['tamanos_base'])  # Creacion del modelo Yolo segun el fichero de configuracion

yolo.cargar_pesos(ruta_pesos)  # Cargado de pesos de red Yolo previamente entrenada

capturador_video = cv2.VideoCapture('/app/Testing/prueba1.mp4')  # apertura de video a procesar
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # configuracion de codec de video
videofps = capturador_video.get(cv2.CAP_PROP_FPS)
ret, frame = capturador_video.read()  # Lectura del primer frame
video_salida = cv2.VideoWriter('/app/Testing/LOCAL/output/salida.mp4', fourcc, videofps,
                               (frame.shape[0], frame.shape[0]))  # Creacion de video de salida procesado con Yolo
numero_frames = int(capturador_video.get(cv2.CAP_PROP_FRAME_COUNT))
print('El numero de frames del video es=' + str(numero_frames))
cv2.namedWindow('test', cv2.WINDOW_NORMAL)#creacion de ventana de reproduccion
frame_actual = 0
while (frame_actual < numero_frames - 10):  # bucle de lectura

    ret, frame = capturador_video.read()  # lectura de frames

    frame = frame[:,
            int(frame.shape[1] / 2) - int(frame.shape[0] / 2):int(frame.shape[1] / 2) + int(frame.shape[0] / 2), :]
    test = frame.copy()

    start = time.time()  # comienzo de medicion de tiempo
    [cajas, caracteristicas] = yolo.predecir(test, 0.25, 0.4)  # prediccion de yolo
    end = time.time()  # finalizacion de medicion de tiempo
    dif = ((end - start))  # diferencia de tiempos
    fps = 1 / dif  # conversion a frames por segundo

    imageaux = dibujar_deteccion(test, cajas,
                                 configuracion['model']['etiquetas'])  # dibujado de detecciones de yolo en la imagen
    video_salida.write(test)  # grabacion de frames en el video de salida
    cv2.imshow('test',test)#visualizacion de frames
    cv2.waitKey(1)#retardo de 1ms necesario para la reproduccion de opencv
    if (frame_actual % 120 == 0):
        print('Frame actual=' + str(frame_actual))
        print('Frames por segundo=' + str(fps))


    frame_actual = frame_actual + 1

video_salida.release()  # finalizacion de video de salida
