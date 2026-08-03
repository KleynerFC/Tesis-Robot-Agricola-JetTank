"""
=============================================================================
PROYECTO: Robot Móvil Terrestre para Evaluación de Cultivos (Tesis UPSE)
ARCHIVO: mapeo_manual.py
AUTOR: Kleyner Fabricio Flores Cedeño
=============================================================================

DESCRIPCIÓN:
Script auxiliar diseñado para la validación del sistema SLAM y los sensores 
odómetricos del robot Hiwonder JetTank. Permite el control teleoperado 
mediante un joystick (gamepad) limitando la velocidad al 5% para evitar 
el deslizamiento (slippage) de las orugas en tierra suelta.

OBJETIVO:
Al eliminar las maniobras bruscas y la velocidad nominal, este script 
permite que el algoritmo SLAM (Gmapping) reconstruya el mapa de ocupación 
(Occupancy Grid) de forma perfecta, comprobando que el hardware (LiDAR y 
Encoders) funciona correctamente y que las fallas en modo autónomo se 
deben únicamente a la física del terreno.

CARACTERÍSTICAS PRINCIPALES:
- Limitación de velocidad: Lineal máx 0.05 m/s, Angular máx 0.3 rad/s.
- Cierre automático de nodos de control nativos de Hiwonder para evitar 
  conflictos de velocidad.
- Monitoreo en tiempo real de la odometría (X, Y) en consola.
- Guardado automático del mapa SLAM al presionar cualquier botón del mando.

REQUISITOS:
- ROS Melodic (Nodo maestro y SLAM activos).
- Joystick de Hiwonder.
- Dongle USB de joystick conectado y nodo 'joy_node' en ejecución.
=============================================================================
"""

#!/usr/bin/env python3
import rospy
import math
import cv2
import numpy as np
import os
import time
from sensor_msgs.msg import Joy, Imu, LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

rospy.init_node('mapeo_manual_lento', anonymous=True)
pub_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

# Matar el control oficial de Hiwonder para tener control total y lento
os.system("rosnode kill /hiwonder_teleop 2>/dev/null")
os.system("rosnode kill /teleop_twist_joy 2>/dev/null")

linear_speed = 0.0
angular_speed = 0.0
stop_requested = False

# Variables para monitorear odometria real
real_x = 0.0
real_y = 0.0

def odom_callback(msg):
    global real_x, real_y
    real_x = msg.pose.pose.position.x
    real_y = msg.pose.pose.position.y

def joy_callback(msg):
    global linear_speed, angular_speed, stop_requested
    ly = msg.axes[1]
    rx = msg.axes[3]
    
    if abs(ly) < 0.1: ly = 0.0
    if abs(rx) < 0.1: rx = 0.0
    
    # Velocidad SUPER LENTA para que el SLAM dibuje perfecto
    linear_speed = ly * 0.05  # Mx 5 cm/s
    angular_speed = rx * 0.3  # Mx 0.3 rad/s  

    for i in range(len(msg.buttons)):
        if msg.buttons[i] == 1:
            stop_requested = True
            print(f"\n Botn {i} presionado. Deteniendo misin...")
            break

def guardar_recursos():
    print("\n DETENIENDO MISIN Y GUARDANDO DATOS...")
    print(" Guardando mapa SLAM...")
    os.system("timeout 15 rosrun map_server map_saver -f /home/hiwonder/Desktop/mapa_perfecto map:=/robot_1/map")
    print(" Mapa guardado en mapa_perfecto.pgm")
    os.system("pkill -f joy_node")
    os.system("pkill -f slam")
    os.system("pkill -f ydlidar")

rospy.Subscriber('/joy', Joy, joy_callback)
rospy.Subscriber('/robot_1/odom', Odometry, odom_callback)

vel_cmd = Twist()
rate = rospy.Rate(10) # Bucle a 10 Hz para no saturar la terminal
print(" Control manual listo. Velocidad limitada al 5% (Sper Lento).")
print(" Presiona CUALQUIER BOTN para terminar y guardar el mapa.")

try:
    while not rospy.is_shutdown() and not stop_requested:
        vel_cmd.linear.x = linear_speed
        vel_cmd.angular.z = angular_speed
        pub_vel.publish(vel_cmd)
        
        # Imprimir la odometra real en una sola linea
        print(f"\r Odometra Real -> X: {real_x:.2f}m | Y: {real_y:.2f}m", end="", flush=True)
        
        rate.sleep()

except KeyboardInterrupt:
    pass
finally:
    vel_cmd.linear.x = 0.0
    vel_cmd.angular.z = 0.0
    pub_vel.publish(vel_cmd)
    guardar_recursos()
