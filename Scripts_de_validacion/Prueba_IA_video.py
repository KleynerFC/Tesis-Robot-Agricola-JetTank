"""
=============================================================================
PROYECTO: Robot Móvil Terrestre para Evaluación de Cultivos (Tesis UPSE)
ARCHIVO: video_ia.py
AUTOR: Kleyner Fabricio Flores Cedeño
=============================================================================

DESCRIPCIÓN:
Script de validación de transmisión de video. Lee un archivo de video, 
ejecuta la inferencia de YOLOv8 en cada cuadro usando TensorRT, y 
transmite el resultado en tiempo real mediante un servidor web MJPEG.

IMPORTANCIA PARA LA TESIS:
Este script fue utilizado para evaluar la viabilidad de transmitir video 
procesado remotamente durante la fase de depuración. Permitió medir el 
consumo de ancho de banda y la tasa de fotogramas (FPS) que la Jetson 
Nano podía sostener al hacer streaming de inferencia. Los resultados 
demostraron que la transmisión continua saturaba el enlace Wi-Fi, lo que 
justificó el diseño de una Interfaz Gráfica (GUI) local en Tkinter y la 
extracción de capturas asíncronas por SFTP en el sistema final.

NOTA:
Diseñado para el modelo multiclase (3 clases). Requiere los archivos 
'best.engine' y un video de prueba (ej. 'cultivo_prueba.mp4') en la 
misma carpeta. Accede desde un navegador a http://<IP_DEL_ROBOT>:8085
=============================================================================
"""

import os
import cv2
import sys
import time
import threading
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

# ==============================================================================
# CONFIGURACION GENERAL
# ==============================================================================
ENGINE_PATH = "best.engine"
VIDEO_PATH = "cultivo_prueba.mp4"  # Nombre de tu archivo de video
PORT = 8085

CLASS_NAMES = {0: "Sana", 1: "Plaga", 2: "Estres"}
CLASS_COLORS = {0: (0, 255, 0), 1: (0, 0, 255), 2: (0, 165, 255)} # Verde, Rojo, Naranja
SCORE_THRESHOLD = 0.45

output_frame = None
frame_lock = threading.Lock()

# ==============================================================================
# SERVIDOR WEB DE STREAMING (MJPEG)
# ==============================================================================
class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global output_frame, frame_lock
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            while True:
                with frame_lock:
                    if output_frame is None:
                        time.sleep(0.01)
                        continue
                    ret, encoded_jpeg = cv2.imencode('.jpg', output_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if not ret:
                        continue
                    frame_bytes = encoded_jpeg.tobytes()
                
                try:
                    self.wfile.write(b'--frame\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(frame_bytes)))
                    self.end_headers()
                    self.wfile.write(frame_bytes)
                    self.wfile.write(b'\r\n')
                    time.sleep(0.04)  # Control de tasa para visualizacion estable
                except (ConnectionResetError, BrokenPipeError):
                    break

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    pass

# ==============================================================================
# PIPELINE DE PROCESAMIENTO DE VIDEO
# ==============================================================================
def main():
    global output_frame, frame_lock

    print("Inicializando entorno de TensorRT...")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    with open(ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # Ubicar memoria en la GPU
    inputs, outputs, allocations = [], [], []
    for binding in engine:
        shape = engine.get_binding_shape(binding)
        size = trt.volume(shape)
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        allocation = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
        allocations.append(int(allocation))
        binding_dict = {'shape': shape, 'dtype': dtype, 'allocation': allocation, 'size': size}
        if engine.binding_is_input(binding):
            inputs.append(binding_dict)
        else:
            outputs.append(binding_dict)

    print(f"Abriendo archivo de video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"Error: No se pudo abrir el archivo de video {VIDEO_PATH}")
        return

    print(f"Simulacion activa. Abre en tu laptop: http://0.0.0.0:{PORT}")
    
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        
        # BUCLE INFINITO: Si el video termina, reinicia el puntero al inicio
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        h_orig, w_orig, _ = frame.shape

        # Preprocesamiento de la imagen (512x512 para el nuevo engine)
        input_img = cv2.resize(frame, (512, 512))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = input_img.astype(np.float32) / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_img = np.ascontiguousarray(np.expand_dims(input_img, axis=0))

        # Computo paralelo en GPU
        cuda.memcpy_htod(inputs[0]['allocation'], input_img)
        context.execute_v2(allocations)
        
        # Recuperar predicciones
        output_data = np.zeros(outputs[0]['shape'], dtype=outputs[0]['dtype'])
        cuda.memcpy_dtoh(output_data, outputs[0]['allocation'])

        # Decodificacion de coordenadas YOLOv8 Multiclase
        output = np.squeeze(output_data).T
        boxes, confidences, class_ids = [], [], []

        for row in output:
            scores = row[4:]
            if len(scores) == 0:
                continue
                
            max_score = np.max(scores)
            if max_score > SCORE_THRESHOLD:
                cls = int(np.argmax(scores))
                xc, yc, w, h = row[0], row[1], row[2], row[3]
                
                x1 = int((xc - w / 2) * (w_orig / 512.0))
                y1 = int((yc - h / 2) * (h_orig / 512.0))
                w_box = int(w * (w_orig / 512.0))
                h_box = int(h * (h_orig / 512.0))
                
                boxes.append([x1, y1, w_box, h_box])
                confidences.append(float(max_score))
                class_ids.append(cls)

        # NMS para mitigar las dobles detecciones por cercania espacial
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=SCORE_THRESHOLD, nms_threshold=0.5)

        # Pintar recuadros de inferencia
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                conf = confidences[i]
                cls_id = class_ids[i]
                
                color = CLASS_COLORS.get(cls_id, (255, 255, 255))
                label = f"{CLASS_NAMES.get(cls_id, 'Desconocido')}: {conf*100:.1f}%"
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Medir la velocidad de procesamiento local del hardware
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS de Inferencia: {fps:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        with frame_lock:
            output_frame = frame.copy()

    cap.release()

if __name__ == '__main__':
    server_address = ('', PORT)
    server = ThreadedHTTPServer(server_address, StreamingHandler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    try:
        main()
    except KeyboardInterrupt:
        print("\nSimulacion por video finalizada.")
        sys.exit(0)
