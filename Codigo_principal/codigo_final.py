"""
=============================================================================
PROYECTO: Robot Móvil Terrestre para Evaluación de Cultivos (Tesis UPSE)
ARCHIVO: codigo_final.py
AUTOR: Kleyner Fabricio Flores Cedeño
=============================================================================

DESCRIPCIÓN:
Este script es el cerebro principal del robot Hiwonder JetTank. Controla la 
navegación autónoma por surcos de cultivo, la clasificación de plantas 
mediante Inteligencia Artificial y la interfaz gráfica de monitoreo.

ARQUITECTURA:
- Navegación: Control PD reactivo usando sensor LiDAR y supervivencia por IMU.
- Visión: Inferencia local con YOLOv8 (TensorRT) para conteo multiclase.
- GUI: Interfaz Tkinter embebida para visualización en tiempo real.

CARACTERÍSTICAS PRINCIPALES:
- Conteo tipo "tráfico" con enfriamiento temporal (cooldown) de 2 segundos.
- Filtro de cercanía espacial para ignorar maleza o hileras lejanas.
- Movimiento automático de cámara PTZ (45°) para inspección lateral.
- Giros en U direccionales (175°) compensando la inercia de las orugas.
- Guardado automático de mapa SLAM y trayectoria al finalizar la misión.

REQUISITOS:
- ROS Melodic (Nodo maestro y SLAM activos)
- Python 3 (OpenCV, PyCUDA, TensorRT, Tkinter)
- Motor de IA compilado: best.engine (TensorRT)
=============================================================================
"""

#!/usr/bin/env python3
import os
import sys
import time
import math
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import tkinter as tk

from collections import deque

# Librerías nativas de ROS 1
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Imu


# =====================================================
# CONFIGURACIÓN DE NAVEGACIÓN, VISIÓN Y SEGURIDAD
# =====================================================

ENGINE_PATH = "/home/hiwonder/jettank_ws/src/src/jettank_control/jettank_control/best.engine"
CAMERA_INDEX = 0

VEL_LINEAL_BUSQUEDA = 0.12
VEL_ANGULAR_MAX = 0.4
VEL_GIRO_U = 0.35
VEL_PIVOTE = 0.06

# Índice central del LiDAR
IDX_CENTRO = 377

# Control PD lateral
KP_PARED = 3.0
KD_PARED = 1.0

# Distancia mínima válida del LiDAR.
UMBRAL_CHASIS = 0.10

# Distancia máxima considerada para paredes/obstáculos.
DIST_MAXIMA_LIDAR = 2.0

# Distancia frontal para detectar fin de fila en última hilera.
# Se deja por compatibilidad, pero la parada final usa los umbrales de misión.
DIST_FIN_FILA_FRONTAL = 0.70

# =====================================================
# UMBRALES DE DETENCIÓN POR BLOQUEO
# =====================================================

# Distancia frontal para considerar bloqueo real durante navegación.
DIST_FRENTE_BLOQUEO = 0.45

# Distancia lateral para considerar bloqueo real durante navegación.
DIST_LATERAL_BLOQUEO = 0.85

# Distancia frontal para detenerse al final de la misión.
DIST_FRENTE_FIN_MISION = 0.70

# Distancia lateral para detenerse al final de la misión.
DIST_LATERAL_FIN_MISION = 1.50

# Distancia máxima para considerar que existe una pared lateral visible.
DIST_PARED_VISIBLE = 1.50

# Distancia deseada a una pared cuando solo se ve una pared.
DIST_OBJETIVO_LATERAL = 0.35

# =====================================================
# DETECCIÓN HÍBRIDA DE FIN DE FILA
# =====================================================

# Distancia a partir de la cual consideramos que una pared lateral está ausente.
DIST_PARED_AUSENTE = 1.80

# Diferencia mínima entre la pared que se abre y la pared presente.
DIFERENCIA_APERTURA = 0.20

# Ángulo de yaw necesario para confirmar que el robot empezó a girar.
ANGULO_ACTIVACION_GIRO = math.radians(15)

# Tiempo mínimo de apertura antes de aceptar yaw como válido.
TIEMPO_APERTURA_MINIMA = 0.35

# Tiempo mínimo de yaw en dirección correcta.
TIEMPO_YAW_MINIMO = 0.15

# Tiempo de confirmación para parada final / fin de fila.
TIEMPO_CONFIRMACION_FIN_FILA = 2.0

# Velocidad angular cuando se detecta apertura y el robot empieza a fugarse.
VEL_APERTURA = 0.40

# Velocidad lineal mientras empieza a fugarse hacia el lado abierto.
VEL_APERTURA_LINEAL = VEL_LINEAL_BUSQUEDA * 0.65

# IMU
UMBRAL_INCLINACION = math.radians(15)
MAX_DESVIACION_YAW = math.radians(30)

# Compensación de inercia en giro U
ANGULO_OBJETIVO = math.radians(175)

# Visión
SCORE_MINIMO = 0.45
TIEMPO_ENFRIAMIENTO = 4.0
Y_LIMIT_INF = 240

# Ángulos cámara PTZ
CAM_ANG_DER = 0.785
CAM_ANG_IZQ = -0.785

# =====================================================
# TIEMPOS DE SEGURIDAD
# =====================================================

# Tiempo que debe mantenerse bloqueo frontal + laterales para detenerse.
TIEMPO_BLOQUEO = 2.0

# Tiempo mínimo que debe haber estado en una fila antes de aceptar fin de fila.
TIEMPO_MINIMO_EN_FILA = 3.0

# Si se quiere que la misión pare al terminar todas las hileras, dejar True.
# Si se quiere modo estricto: única parada por bloqueo frontal/lateral,
# poner False. En ese caso el robot seguirá hasta bloqueo, cerrar táctilmente la terminal o Ctrl+C.
PARADA_POR_FIN_MISION = True


# =======================================
# 1. INICIALIZAR LA CÁMARA PRIMERO CON V4L2
# =======================================
print("Iniciando camara con backend V4L2...")
cap = cv2.VideoCapture(CAMERA_INDEX + cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("ERROR: No se pudo abrir la camara. Abortando.")
    sys.exit(1)

print("Camara lista.")


# =================================
# 2. INICIALIZACIÓN DE MOTOR TENSORRT
# =================================
print("Cargando TensorRT para conteo y clasificacion...")
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

with open(ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

inputs, outputs, allocations = [], [], []

for binding in engine:
    shape = engine.get_binding_shape(binding)
    size = trt.volume(shape)
    dtype = trt.nptype(engine.get_binding_dtype(binding))
    allocation = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
    allocations.append(int(allocation))

    binding_dict = {
        'shape': shape,
        'dtype': dtype,
        'allocation': allocation
    }

    if engine.binding_is_input(binding):
        inputs.append(binding_dict)
    else:
        outputs.append(binding_dict)

print("Esperando 5s para que el bus USB y el LiDAR se recuperen...")
time.sleep(5.0)


# =========================================
# CONFIGURACIÓN DEL NODO ROS Y SENSORES
# =========================================
rospy.init_node('navegacion_lidar_cultivos', anonymous=True)

pub_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
pub_pan = rospy.Publisher('/joint1_controller/command', Float64, queue_size=1)
pub_tilt = rospy.Publisher('/joint2_controller/command', Float64, queue_size=1)

vel_cmd = Twist()

# Odometría
current_yaw = 0.0
odom_received = False
trayectoria_puntos = []
pos_x_calc = 0.0
pos_y_calc = 0.0


def calcular_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def odom_callback(msg):
    global current_yaw, odom_received, pos_x_calc, pos_y_calc
    current_yaw = calcular_yaw(msg.pose.pose.orientation)
    odom_received = True
    pos_x_calc = msg.pose.pose.position.x
    pos_y_calc = msg.pose.pose.position.y


# IMU
current_pitch = 0.0
current_roll = 0.0
imu_received = False


def imu_callback(msg):
    global current_pitch, current_roll, imu_received

    q = msg.orientation

    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    if abs(sinp) >= 1:
        current_pitch = math.copysign(math.pi / 2, sinp)
    else:
        current_pitch = math.asin(sinp)

    sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
    cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    current_roll = math.atan2(sinr_cosp, cosr_cosp)

    imu_received = True


# LiDAR
dist_izq = 10.0
dist_der = 10.0
dist_frontal = 10.0
scan_received = False

# Historiales para suavizar lecturas LiDAR.
hist_frontal = deque(maxlen=3)
hist_izq = deque(maxlen=3)
hist_der = deque(maxlen=3)


def distancia_robusta(vals, percentil=15):
    """
    Devuelve una distancia robusta frente a ruido.
    """
    limpio = []

    for r in vals:
        if math.isfinite(r) and UMBRAL_CHASIS < r < DIST_MAXIMA_LIDAR:
            limpio.append(r)

    if len(limpio) == 0:
        return 10.0

    limpio.sort()

    if len(limpio) <= 2:
        return limpio[0]

    idx = int(round((percentil / 100.0) * (len(limpio) - 1)))
    return limpio[idx]


def scan_callback(msg):
    global dist_izq, dist_der, dist_frontal, scan_received

    puntos = len(msg.ranges)

    def sector(start_idx, end_idx):
        return [msg.ranges[i % puntos] for i in range(start_idx, end_idx)]

    f_raw = distancia_robusta(sector(IDX_CENTRO - 25, IDX_CENTRO + 25), 10)
    l_raw = distancia_robusta(sector(IDX_CENTRO + 40, IDX_CENTRO + 100), 15)
    r_raw = distancia_robusta(sector(IDX_CENTRO - 100, IDX_CENTRO - 40), 15)

    hist_frontal.append(f_raw)
    hist_izq.append(l_raw)
    hist_der.append(r_raw)

    dist_frontal = float(np.median(hist_frontal))
    dist_izq = float(np.median(hist_izq))
    dist_der = float(np.median(hist_der))

    scan_received = True


rospy.Subscriber('/robot_1/odom', Odometry, odom_callback)
rospy.Subscriber('/robot_1/imu', Imu, imu_callback)
rospy.Subscriber('/scan', LaserScan, scan_callback)

rospy.sleep(0.6)

pub_pan.publish(Float64(CAM_ANG_DER))
pub_tilt.publish(Float64(0.0))
print("Mecanismo PTZ calibrado. Camara apuntando a la derecha (Carril 1).")


# ============================================
# 3. INTERFAZ GRÁFICA (GUI)
# ============================================
root = tk.Tk()
root.title("Panel de Monitoreo - Tesis JetTank")
root.geometry("480x320")
root.configure(bg="black")

estado_txt = tk.StringVar(value="Iniciando...")
total_txt = tk.StringVar(value="0")
sanas_txt = tk.StringVar(value="0")
plagas_txt = tk.StringVar(value="0")
estres_txt = tk.StringVar(value="0")
pos_txt = tk.StringVar(value="Esperando odometria...")

tk.Label(
    root,
    text="ESTADO DEL ROBOT",
    font=("Helvetica", 12, "bold"),
    bg="black",
    fg="white"
).pack(pady=5)

lbl_estado = tk.Label(
    root,
    textvariable=estado_txt,
    font=("Helvetica", 16),
    bg="black",
    fg="blue"
)
lbl_estado.pack()

tk.Label(
    root,
    text="PLANTAS DETECTADAS",
    font=("Helvetica", 12, "bold"),
    bg="black",
    fg="white"
).pack(pady=5)

tk.Label(
    root,
    textvariable=total_txt,
    font=("Helvetica", 36, "bold"),
    bg="black",
    fg="cyan"
).pack()

f_counts = tk.Frame(root, bg="black")
f_counts.pack(pady=5)

tk.Label(f_counts, text="Sanas:", font=("Helvetica", 10), bg="black", fg="green").grid(row=0, column=0, padx=5)
tk.Label(f_counts, textvariable=sanas_txt, font=("Helvetica", 10, "bold"), bg="black", fg="green").grid(row=0, column=1, padx=5)

tk.Label(f_counts, text="Plaga:", font=("Helvetica", 10), bg="black", fg="red").grid(row=0, column=2, padx=5)
tk.Label(f_counts, textvariable=plagas_txt, font=("Helvetica", 10, "bold"), bg="black", fg="red").grid(row=0, column=3, padx=5)

tk.Label(f_counts, text="Estres:", font=("Helvetica", 10), bg="black", fg="orange").grid(row=0, column=4, padx=5)
tk.Label(f_counts, textvariable=estres_txt, font=("Helvetica", 10, "bold"), bg="black", fg="orange").grid(row=0, column=5, padx=5)

tk.Label(root, textvariable=pos_txt, font=("Helvetica", 10), bg="black", fg="gray").pack(side=tk.BOTTOM, pady=5)

root.update()


# =======================================
# FUNCIONES DE EMERGENCIA, LIMPIEZA Y GRAFICADO
# =======================================
def apagar_sistema_seguro():
    print("Mision finalizada. Deteniendo motores...")

    vel_cmd.linear.x = 0.0
    vel_cmd.angular.z = 0.0
    pub_vel.publish(vel_cmd)

    pub_pan.publish(Float64(0.0))
    print("Camara centrada.")

    if len(trayectoria_puntos) > 2:
        print("Dibujando y guardando trayectoria...")

        img = np.zeros((500, 500, 3), dtype=np.uint8)

        xs = [p[0] for p in trayectoria_puntos]
        ys = [p[1] for p in trayectoria_puntos]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        scale = 400 / max(max_x - min_x + 0.1, max_y - min_y + 0.1)

        pts = []
        for x, y in trayectoria_puntos:
            px = int(50 + (x - min_x) * scale)
            py = int(450 - (y - min_y) * scale)
            pts.append([px, py])

        pts_np = np.array(pts, np.int32)

        cv2.polylines(img, [pts_np], False, (255, 0, 0), 2)
        cv2.circle(img, pts[0], 6, (0, 255, 0), -1)
        cv2.circle(img, pts[-1], 6, (0, 0, 255), -1)
        cv2.putText(img, "Trayectoria del Robot", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imwrite('/home/hiwonder/Desktop/trayectoria.png', img)
        print("Trayectoria guardada.")

    print("Guardando mapa SLAM...")
    os.system("timeout 15 rosrun map_server map_saver -f /home/hiwonder/Desktop/mapa_cultivo map:=/robot_1/map")

    print("Cerrando SLAM/LIDAR...")
    os.system("pkill -f slam")
    os.system("pkill -f ydlidar")
    os.system("pkill -f gmapping")
    os.system("pkill -f hector")
    os.system("pkill -f map_saver")

    estado_txt.set("MISION COMPLETADA")
    lbl_estado.config(fg="green")
    pos_txt.set("SISTEMA DETENIDO. TOMA LA FOTO. CERRANDO EN 30s...")
    root.update()

    print("Toma la foto de la interfaz! Cerrando en 30 segundos...")
    time.sleep(30.0)

    try:
        root.destroy()
    except:
        pass


# =======================================
# TIMEOUT DE SENSORES
# =======================================
print("Esperando Odometria, IMU y LiDAR (Timeout 15s)...")

timeout_start = time.time()

while not rospy.is_shutdown() and (not odom_received or not scan_received or not imu_received) and (time.time() - timeout_start) < 15.0:
    time.sleep(0.5)

if not odom_received or not scan_received or not imu_received:
    print("ERROR: Faltan sensores. Abortando script.")
    apagar_sistema_seguro()
    sys.exit(1)
else:
    print("Sensores listos. LiDAR, IMU y Odometria operativos!")


# ====================================
# MÁQUINA DE ESTADOS
# ====================================
estado_actual = "NAVEGANDO_FILA"
hila_actual = 1
total_hileras = 3

yaw_inicio_fila = 0.0
objetivo_yaw_giro = 0.0
direccion_giro = 1
tolerancia_yaw = 0.15

error_previo = 0.0

contador_plantas = 0
sanas = 0
plagas = 0
estres = 0

tiempo_ultimo_conteo = 0.0
deteccion_activada = True

# Timers de seguridad
t_bloqueo = 0.0
t_apertura = 0.0
t_yaw_giro = 0.0
t_inicio_fila = time.time()
t_entrada_fila = 0.0

# Bandera para modo estricto si PARADA_POR_FIN_MISION = False
mision_finalizada = False


def normalizar_angulo(angulo):
    while angulo > math.pi:
        angulo -= 2 * math.pi
    while angulo < -math.pi:
        angulo += 2 * math.pi
    return angulo


yaw_inicio_fila = current_yaw
t_inicio_fila = time.time()

print("JetTank listo! Iniciando recorrido...")

rate = rospy.Rate(30)


# ============================
# BUCLE PRINCIPAL
# ============================
try:
    while not rospy.is_shutdown() and cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        h_orig, w_orig, _ = frame.shape

        # ============================
        # INFERENCIA VISUAL
        # ============================
        if deteccion_activada:
            input_img = cv2.resize(frame, (512, 512))
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            input_img = input_img.transpose(2, 0, 1)
            input_img = np.ascontiguousarray(np.expand_dims(input_img, axis=0))

            cuda.memcpy_htod(inputs[0]['allocation'], input_img)
            context.execute_v2(allocations)

            output_data = np.zeros(outputs[0]['shape'], dtype=outputs[0]['dtype'])
            cuda.memcpy_dtoh(output_data, outputs[0]['allocation'])

            output = np.squeeze(output_data).T

            deteccion_actual_valida = False
            clase_actual = -1
            max_score_actual = 0.0

            for row in output:
                scores = row[4:]

                if len(scores) == 0:
                    continue

                max_score = np.max(scores)

                if max_score > SCORE_MINIMO and max_score > max_score_actual:
                    cls = int(np.argmax(scores))

                    xc, w = row[0], row[2]
                    yc, h = row[1], row[3]

                    y1 = int((yc - h / 2) * (h_orig / 512.0))
                    target_h = int(h * (h_orig / 512.0))
                    y_bottom = y1 + target_h

                    if y_bottom > Y_LIMIT_INF:
                        max_score_actual = max_score
                        deteccion_actual_valida = True
                        clase_actual = cls

            if deteccion_actual_valida:
                if time.time() - tiempo_ultimo_conteo > TIEMPO_ENFRIAMIENTO:
                    contador_plantas += 1
                    tiempo_ultimo_conteo = time.time()

                    if clase_actual == 0:
                        sanas += 1
                    elif clase_actual == 1:
                        plagas += 1
                    elif clase_actual == 2:
                        estres += 1

                    print(f"Planta {contador_plantas} detectada! Clase: {clase_actual} (Score: {max_score_actual:.2f})")

        # ============================
        # SEGURIDAD Y CONDICIONES BÁSICAS
        # ============================
        esta_escalando = abs(current_pitch) > UMBRAL_INCLINACION or abs(current_roll) > math.radians(10)
        t_now = time.time()

        yaw_relativo = normalizar_angulo(current_yaw - yaw_inicio_fila)

        # Visibilidad de paredes
        izq_visible = dist_izq < DIST_PARED_VISIBLE
        der_visible = dist_der < DIST_PARED_VISIBLE
        paredes_visibles = izq_visible or der_visible

        # =====================================
        # BLOQUEO REAL: FRONTAL + LATERALES
        # =====================================
        frontal_cerca = dist_frontal < DIST_FRENTE_BLOQUEO
        izq_cerca = dist_izq < DIST_LATERAL_BLOQUEO
        der_cerca = dist_der < DIST_LATERAL_BLOQUEO

        bloqueado = frontal_cerca and izq_cerca and der_cerca

        if bloqueado:
            if t_bloqueo == 0.0:
                t_bloqueo = t_now

            if (t_now - t_bloqueo) >= TIEMPO_BLOQUEO:
                print("BLOQUEO CONFIRMADO: pared frontal y laterales por más de 2 segundos.")
                print("Deteniendo el robot de forma segura.")

                vel_cmd.linear.x = 0.0
                vel_cmd.angular.z = 0.0
                pub_vel.publish(vel_cmd)

                break
        else:
            t_bloqueo = 0.0

        # =====================================
        # DETECCIÓN HÍBRIDA DE FIN DE FILA
        # =====================================
        apertura_instantanea = False
        apertura_confirmada = False
        yaw_en_direccion = False
        yaw_giro_confirmado = False
        giro_esperado = 0
        fin_fila_confirmada = False

        if estado_actual == "NAVEGANDO_FILA" and not mision_finalizada:

            if hila_actual == 1:
                # Hilera 1: giro a la derecha.
                # La pared derecha debe abrirse respecto a la izquierda.
                giro_esperado = -1

                apertura_instantanea = (
                    izq_visible and
                    (
                        dist_der > dist_izq + DIFERENCIA_APERTURA or
                        (dist_der > DIST_PARED_AUSENTE and dist_der > dist_izq + 0.10)
                    )
                )

                yaw_en_direccion = yaw_relativo < -ANGULO_ACTIVACION_GIRO

            elif hila_actual == 2:
                # Hilera 2: giro a la izquierda.
                # La pared izquierda debe abrirse respecto a la derecha.
                giro_esperado = 1

                apertura_instantanea = (
                    der_visible and
                    (
                        dist_izq > dist_der + DIFERENCIA_APERTURA or
                        (dist_izq > DIST_PARED_AUSENTE and dist_izq > dist_der + 0.10)
                    )
                )

                yaw_en_direccion = yaw_relativo > ANGULO_ACTIVACION_GIRO

            else:
                # Última hilera: no debe hacer otro giro en U.
                # Debe detenerse si tiene las 3 paredes durante 2 segundos.
                giro_esperado = 0

                pared_fin_mision = (
                    dist_frontal < DIST_FRENTE_FIN_MISION and
                    dist_izq < DIST_LATERAL_FIN_MISION and
                    dist_der < DIST_LATERAL_FIN_MISION
                )

                apertura_instantanea = pared_fin_mision
                yaw_en_direccion = False

            if apertura_instantanea and not bloqueado:
                if t_apertura == 0.0:
                    t_apertura = t_now

                apertura_confirmada = (t_now - t_apertura) >= TIEMPO_APERTURA_MINIMA

            else:
                t_apertura = 0.0
                apertura_confirmada = False

            if apertura_confirmada and yaw_en_direccion:
                if t_yaw_giro == 0.0:
                    t_yaw_giro = t_now

                yaw_giro_confirmado = (t_now - t_yaw_giro) >= TIEMPO_YAW_MINIMO

            else:
                t_yaw_giro = 0.0
                yaw_giro_confirmado = False

            if apertura_confirmada:
                tiempo_apertura = t_now - t_apertura

                if hila_actual >= total_hileras:
                    fin_fila_confirmada = tiempo_apertura >= TIEMPO_CONFIRMACION_FIN_FILA
                else:
                    # La condición principal es yaw + apertura.
                    # Si el yaw no llega, igualmente se confirma por apertura prolongada.
                    fin_fila_confirmada = (
                        yaw_giro_confirmado or
                        tiempo_apertura >= TIEMPO_CONFIRMACION_FIN_FILA
                    )
            else:
                fin_fila_confirmada = False

            # Descomentar para depurar:
            # print(f"[CHECK] h={hila_actual} YAW={math.degrees(yaw_relativo):.1f} I={dist_izq:.2f} D={dist_der:.2f} F={dist_frontal:.2f} AP={apertura_instantanea} FIN={fin_fila_confirmada}")

        else:
            t_apertura = 0.0
            t_yaw_giro = 0.0
            apertura_confirmada = False
            yaw_giro_confirmado = False
            fin_fila_confirmada = False

        # ============================
        # MÁQUINA DE ESTADOS
        # ============================
        if estado_actual == "NAVEGANDO_FILA":
            deteccion_activada = True

            if esta_escalando:
                t_apertura = 0.0
                t_yaw_giro = 0.0

                vel_cmd.linear.x = -0.06

                if abs(current_roll) > math.radians(5):
                    vel_cmd.angular.z = current_roll * 5.0
                else:
                    vel_cmd.angular.z = 0.0

            elif fin_fila_confirmada and (t_now - t_inicio_fila) >= TIEMPO_MINIMO_EN_FILA and not mision_finalizada:

                if hila_actual >= total_hileras:
                    if PARADA_POR_FIN_MISION:
                        print("Mision Completada. Todas las hileras recorridas.")
                        print("Condición de parada final por 3 paredes confirmada.")

                        vel_cmd.linear.x = 0.0
                        vel_cmd.angular.z = 0.0
                        pub_vel.publish(vel_cmd)

                        break
                    else:
                        print("Mision finalizada, pero PARADA_POR_FIN_MISION=False.")
                        print("El robot no se detiene por fin de misión. Solo parará por bloqueo o Ctrl+C.")
                        mision_finalizada = True

                        t_apertura = 0.0
                        t_yaw_giro = 0.0

                        vel_cmd.linear.x = VEL_LINEAL_BUSQUEDA
                        vel_cmd.angular.z = 0.0

                else:
                    estado_actual = "PREPARANDO_GIRO"

                    t_apertura = 0.0
                    t_yaw_giro = 0.0

            else:
                # Si hay apertura en la hilera correcta, se deja que el robot
                # empiece a girar hacia el lado abierto.
                if apertura_instantanea and giro_esperado != 0 and not bloqueado:
                    vel_cmd.linear.x = VEL_APERTURA_LINEAL
                    vel_cmd.angular.z = giro_esperado * VEL_APERTURA
                    error_previo = 0.0

                elif paredes_visibles:
                    if izq_visible and der_visible:
                        # Ambas paredes visibles: centrado clásico.
                        error_actual = (dist_izq - dist_der) / 2.0

                    elif izq_visible:
                        # Solo pared izquierda visible.
                        error_actual = dist_izq - DIST_OBJETIVO_LATERAL

                    else:
                        # Solo pared derecha visible.
                        error_actual = DIST_OBJETIVO_LATERAL - dist_der

                    # Filtro simple del error para reducir ruido.
                    error_actual = 0.7 * error_previo + 0.3 * error_actual

                    d_error = error_actual - error_previo
                    error_previo = error_actual

                    vel_cmd.linear.x = VEL_LINEAL_BUSQUEDA
                    vel_cmd.angular.z = (KP_PARED * error_actual) + (KD_PARED * d_error)

                    if vel_cmd.angular.z > VEL_ANGULAR_MAX:
                        vel_cmd.angular.z = VEL_ANGULAR_MAX

                    if vel_cmd.angular.z < -VEL_ANGULAR_MAX:
                        vel_cmd.angular.z = -VEL_ANGULAR_MAX

                else:
                    # Sin paredes: seguir recto, no detenerse.
                    vel_cmd.linear.x = VEL_LINEAL_BUSQUEDA
                    vel_cmd.angular.z = 0.0
                    error_previo = 0.0

        elif estado_actual == "PREPARANDO_GIRO":
            deteccion_activada = False

            # No se coloca velocidad cero aquí para evitar pausas falsas.

            if hila_actual == 1:
                direccion_giro = -1
                objetivo_yaw_giro = normalizar_angulo(yaw_inicio_fila - ANGULO_OBJETIVO)
                pub_pan.publish(Float64(CAM_ANG_IZQ))
                print("Camara girada a la izquierda para Carril 2")
                estado_actual = "GIRANDO_U"

            elif hila_actual == 2:
                direccion_giro = 1
                objetivo_yaw_giro = normalizar_angulo(yaw_inicio_fila + ANGULO_OBJETIVO)
                pub_pan.publish(Float64(CAM_ANG_DER))
                print("Camara girada a la derecha para Carril 3")
                estado_actual = "GIRANDO_U"

            else:
                print("Estado de giro no válido. Volviendo a navegación.")
                estado_actual = "NAVEGANDO_FILA"

        elif estado_actual == "GIRANDO_U":
            deteccion_activada = False

            error_yaw = normalizar_angulo(objetivo_yaw_giro - current_yaw)
            angulo_restante = abs(error_yaw)
            angulo_girado = ANGULO_OBJETIVO - angulo_restante

            if esta_escalando:
                vel_cmd.linear.x = -0.05
                vel_cmd.angular.z = direccion_giro * 0.2

            elif angulo_girado < math.radians(160):
                if direccion_giro == -1:
                    pared_interna = dist_der
                else:
                    pared_interna = dist_izq

                if pared_interna < 0.25:
                    vel_cmd.linear.x = -0.05
                    vel_cmd.angular.z = direccion_giro * (VEL_GIRO_U + 0.1)

                elif pared_interna < 0.40:
                    vel_cmd.linear.x = VEL_PIVOTE
                    vel_cmd.angular.z = direccion_giro * (VEL_GIRO_U * 0.8)

                else:
                    vel_cmd.linear.x = VEL_PIVOTE
                    vel_cmd.angular.z = direccion_giro * VEL_GIRO_U

            elif angulo_restante > tolerancia_yaw:
                vel_cmd.linear.x = VEL_PIVOTE
                vel_cmd.angular.z = direccion_giro * max(0.15, min(VEL_GIRO_U, angulo_restante * 1.5))

            else:
                # Giro terminado.
                hila_actual += 1

                yaw_inicio_fila = current_yaw
                t_entrada_fila = time.time()

                estado_actual = "ENTRANDO_FILA"

                vel_cmd.linear.x = VEL_LINEAL_BUSQUEDA
                vel_cmd.angular.z = 0.0

                print(f"Giro completado. Entrando a la hilera {hila_actual}")

        elif estado_actual == "ENTRANDO_FILA":
            deteccion_activada = True

            # Avanzar recto durante unos segundos para entrar a la nueva hilera.
            vel_cmd.linear.x = VEL_LINEAL_BUSQUEDA
            vel_cmd.angular.z = 0.0

            if (t_now - t_entrada_fila) >= 2.0:
                estado_actual = "NAVEGANDO_FILA"

                # Reiniciar referencia de yaw y tiempo de fila después de entrar.
                yaw_inicio_fila = current_yaw
                t_inicio_fila = time.time()

                error_previo = 0.0
                t_apertura = 0.0
                t_yaw_giro = 0.0

                print("Entrada a nueva hilera completada. Navegando.")

        # Publicar velocidad
        pub_vel.publish(vel_cmd)

        # Trayectoria
        trayectoria_puntos.append((pos_x_calc, pos_y_calc))

        # ============================
        # ACTUALIZAR GUI
        # ============================
        estado_txt.set(estado_actual)
        total_txt.set(str(contador_plantas))
        sanas_txt.set(str(sanas))
        plagas_txt.set(str(plagas))
        estres_txt.set(str(estres))

        if estado_actual in ["NAVEGANDO_FILA", "ENTRANDO_FILA"]:
            lbl_estado.config(fg="green")
        elif "GIRO" in estado_actual:
            lbl_estado.config(fg="orange")

        if esta_escalando:
            lbl_estado.config(fg="red")
            estado_txt.set("ALERTA ESCALADA")

        if mision_finalizada and not PARADA_POR_FIN_MISION:
            estado_txt.set(f"{estado_actual} (MISION FINALIZADA, SIN PARADA)")

        pos_txt.set(f"Pos Robot: X={pos_x_calc:.1f}m, Y={pos_y_calc:.1f}m")

        root.update_idletasks()
        root.update()

        rate.sleep()

except KeyboardInterrupt:
    print("\nEjecucion interrumpida por el usuario (Ctrl+C).")

finally:
    apagar_sistema_seguro()
    print(f"Resumen final: Total plantas: {contador_plantas} (Sanas: {sanas}, Plaga: {plagas}, Estres: {estres})")
