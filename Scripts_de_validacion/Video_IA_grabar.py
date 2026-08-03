"""
=============================================================================
PROYECTO: Robot Móvil Terrestre para Evaluación de Cultivos (Tesis UPSE)
ARCHIVO: video_ia_grabar.py
AUTOR: Kleyner Fabricio Flores Cedeño
=============================================================================

DESCRIPCIÓN:
Script de validación de inferencia y registro visual. A diferencia de 
'video_ia.py', este script graba el video procesado con las detecciones 
de YOLOv8 en un archivo físico (.avi) en el disco duro, además de 
transmitirlo por streaming.

IMPORTANCIA PARA LA TESIS:
Este script fue utilizado para generar material audiovisual de evidencia 
durante la fase de desarrollo. Al guardar el video con las cajas de 
delimitación y la clasificación multiclase dibujadas, se pudo analizar 
detenidamente el rendimiento del modelo fuera de tiempo real, medir la 
estabilidad de las detecciones y extraer capturas de pantalla para el 
documento de tesis.

NOTA:
Diseñado para el modelo multiclase (3 clases). Requiere los archivos 
'best.engine' y un video de prueba (ej. 'cultivo_prueba.mp4') en la 
misma carpeta.
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
VIDEO_PATH = "cultivo_prueba.mp4"
OUTPUT_VIDEO_PATH = "simulacion_lechugas.avi"  # Video de salida fluido
PORT = 8089

CLASS_NAMES = {0: "Sana", 1: "Plaga", 2: "Estres"}
CLASS_COLORS = {0: (0, 255, 0), 1: (0, 0, 255), 2: (0, 165, 255)} # Verde, Rojo, Naranja
SCORE_THRESHOLD = 0.45

output_frame = None
frame_lock = threading.Lock()

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
                    time.sleep(0.05)
                except (ConnectionResetError, BrokenPipeError):
                    break

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    pass

# ==============================================================================
# PIPELINE DE PROCESAMIENTO Y GRABACION
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
        print(f"Error: No se pudo abrir el video {VIDEO_PATH}")
        return

    # Obtener propiedades del video original para el grabador
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_orig = cap.get(cv2.CAP_PROP_FPS)
    if fps_orig == 0 or fps_orig > 60: 
        fps_orig = 25.0  # Forzar un estandar si no lee los FPS del metraje

    print(f"Configurando grabador de video ({w_orig}x{h_orig} a {fps_orig} FPS)...")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps_orig, (w_orig, h_orig))

    print(f"Procesando y grabando. Puedes monitorear en: http://0.0.0.0:{PORT}")
    
    frame_count = 0
    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            
            # Al terminar el video, rompemos el ciclo para cerrar el archivo correctamente
            if not ret:
                print("Se llego al final del video de prueba.")
                break

            # Preprocesamiento para la GPU (512x512 para el nuevo engine)
            input_img = cv2.resize(frame, (512, 512))
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            input_img = input_img.astype(np.float32) / 255.0
            input_img = input_img.transpose(2, 0, 1)
            input_img = np.ascontiguousarray(np.expand_dims(input_img, axis=0))

            # Inferencia CUDA
            cuda.memcpy_htod(inputs[0]['allocation'], input_img)
            context.execute_v2(allocations)
            
            # Descargar resultados
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

            indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=SCORE_THRESHOLD, nms_threshold=0.5)

            # Pintar recuadros
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    conf = confidences[i]
                    cls_id = class_ids[i]
                    
                    color = CLASS_COLORS.get(cls_id, (255, 255, 255))
                    label = f"{CLASS_NAMES.get(cls_id, 'Desconocido')}: {conf*100:.1f}%"
                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Escribir de forma nativa el fotograma procesado en el archivo de video
            out_video.write(frame)
            frame_count += 1

            # Medicion de velocidad en pantalla
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(frame, f"FPS de Inferencia: {fps:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            with frame_lock:
                output_frame = frame.copy()

    finally:
        # Asegurar el cierre correcto de los objetos para evitar archivos corruptos
        cap.release()
        out_video.release()
        print(f"Exito! Archivo guardado con {frame_count} fotogramas procesados.")

if __name__ == '__main__':
    server_address = ('', PORT)
    server = ThreadedHTTPServer(server_address, StreamingHandler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    try:
        main()
    except KeyboardInterrupt:
        print("\nGrabacion interrumpida manualmente.")
