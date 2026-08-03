"""
=============================================================================
PROYECTO: Robot Móvil Terrestre para Evaluación de Cultivos (Tesis UPSE)
ARCHIVO: prueba_ia.py
AUTOR: Kleyner Fabricio Flores Cedeño
=============================================================================

DESCRIPCIÓN:
Script de validación offline para probar el motor de inferencia TensorRT 
(best.engine) fuera del entorno de ROS. Lee una imagen estática, ejecuta 
la red neuronal YOLOv8, dibuja las cajas de delimitación clasificando 
multiclase (Sana, Plaga, Estrés) y guarda el resultado en el disco.

IMPORTANCIA PARA LA TESIS:
Este script fue fundamental durante la fase de despliegue (Edge Computing). 
Permitió verificar que el pipeline de optimización de PyTorch a TensorRT 
funcionaba correctamente en la GPU Maxwell de la Jetson Nano, midiendo la 
precisión de la detección antes de integrar la inferencia en tiempo real 
dentro del bucle de control reactivo de ROS.

NOTA:
Diseñado para el modelo multiclase (3 clases). Requiere que los archivos 
'best.engine' y una imagen de prueba (ej. 'test_lechuga.jpg') estén en la 
misma carpeta que este script al ejecutarse.
=============================================================================
"""

import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Configuracion de rutas
ENGINE_PATH = "best.engine"
IMAGE_PATH = "test_lechuga.jpg"
OUTPUT_PATH = "resultado_validacion.jpg"

# Definicion de clases y colores (BGR para OpenCV)
CLASS_NAMES = {0: "Sana", 1: "Plaga", 2: "Estres"}
CLASS_COLORS = {0: (0, 255, 0), 1: (0, 0, 255), 2: (0, 165, 255)} # Verde, Rojo, Naranja

SCORE_THRESHOLD = 0.45

print("Inicializando entorno de TensorRT...")
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# 1. Cargar y deserializar el motor .engine
with open(ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# 2. Reservar memoria en la GPU (Buffers)
inputs = []
outputs = []
allocations = []

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

# 3. Cargar y preprocesar la imagen
print(f"Procesando imagen de entrada: {IMAGE_PATH}")
orig_img = cv2.imread(IMAGE_PATH)
if orig_img is None:
    raise FileNotFoundError(f"No se encontro la imagen en {IMAGE_PATH}")

h_orig, w_orig, _ = orig_img.shape

# Ajustar tamano a 512x512 (resolucion del nuevo engine multiclase)
input_img = cv2.resize(orig_img, (512, 512))
input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
input_img = input_img.astype(np.float32) / 255.0
input_img = input_img.transpose(2, 0, 1)
input_img = np.ascontiguousarray(np.expand_dims(input_img, axis=0))

# 4. Ejecutar la inferencia en los nucleos CUDA
print("Lanzando operaciones matematicas en la GPU de la Jetson Nano...")
cuda.memcpy_htod(inputs[0]['allocation'], input_img)
context.execute_v2(allocations)

output_data = np.zeros(outputs[0]['shape'], dtype=outputs[0]['dtype'])
cuda.memcpy_dtoh(output_data, outputs[0]['allocation'])

# 5. Post-procesamiento de la salida de YOLOv8 Multiclase
output = np.squeeze(output_data).T

boxes = []
confidences = []
class_ids = []

print("Filtrando detecciones...")
for row in output:
    scores = row[4:]
    if len(scores) == 0:
        continue
        
    max_score = np.max(scores)
    if max_score > SCORE_THRESHOLD:
        cls = int(np.argmax(scores))
        xc, yc, w, h = row[0], row[1], row[2], row[3]
        
        # Escalar de 512x512 al tamano original de la foto
        x1 = int((xc - w / 2) * (w_orig / 512.0))
        y1 = int((yc - h / 2) * (h_orig / 512.0))
        w_box = int(w * (w_orig / 512.0))
        h_box = int(h * (h_orig / 512.0))
        
        boxes.append([x1, y1, w_box, h_box])
        confidences.append(float(max_score))
        class_ids.append(cls)

# Aplicar Non-Maximum Suppression (NMS)
indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=SCORE_THRESHOLD, nms_threshold=0.5)

# 6. Dibujar resultados
if len(indices) > 0:
    print(f"Exito! Se detectaron {len(indices)} plantas.")
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        conf = confidences[i]
        cls_id = class_ids[i]
        
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        label = f"{CLASS_NAMES.get(cls_id, 'Desconocido')}: {conf*100:.1f}%"
        
        cv2.rectangle(orig_img, (x, y), (x + w, y + h), color, 3)
        cv2.putText(orig_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
else:
    print("Inferencia completada, pero no se reconocio ninguna planta con suficiente confianza.")

cv2.imwrite(OUTPUT_PATH, orig_img)
print(f"Imagen guardada exitosamente como: {OUTPUT_PATH}")
