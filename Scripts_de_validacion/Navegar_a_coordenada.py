"""
=============================================================================
PROYECTO: Robot Móvil Terrestre para Evaluación de Cultivos (Tesis UPSE)
ARCHIVO: odom.py
AUTOR: Kleyner Fabricio Flores Cedeño
=============================================================================

DESCRIPCIÓN:
Script de validación que demuestra el funcionamiento de la odometría pura 
(Dead Reckoning). Aplica una matriz de rotación 2D para establecer un marco 
de referencia global y utiliza un controlador Proporcional (P) para navegar 
hacia una coordenada (X, Y) en metros.

REQUISITOS PREVIOS:

Tener el nodo del SLAM corriendo (mediante el ícono Iniciar SLAM).
Ejecutar el script en un terreno plano y con alta fricción (ej. baldosas, cemento o cerámica).
Advertencia: No mover ni tocar el robot durante los primeros 40 segundos de iniciado el SLAM, 
para permitir la calibración correcta del giroscopio de la IMU.
Configuración de la meta:En el código, dentro de la clase OdometryNav.__init__, se pueden 
modificar los siguientes parámetros:

self.start_yaw_deg: Orientación física inicial del robot al encenderlo (0=Frente, 90=Izquierda, 180=Atrás, 270=Derecha).
self.target_x y self.target_y: Coordenadas de destino en metros desde el punto de arranque.


IMPORTANCIA PARA LA TESIS:
Este experimento comprobó que los encoders magnéticos y la IMU del robot 
están perfectamente calibrados. En superficies planas, el robot logra 
alcanzar la meta con precisión milimétrica.

ADVERTENCIA Y LIMITACIÓN FÍSICA:
Este sistema funciona excelentemente SIEMPRE Y CUANDO el terreno sea plano 
y no sea resbaloso (ej. piso de cemento o baldosas). 
En condiciones de campo (como la tierra suelta del patio de ensayos), las 
orugas sufren deslizamiento (slippage). Esto provoca que los encoders cuenten 
vueltas en falso, generando una deriva acumulativa que hace que el robot 
falle en llegar a la meta. Esta limitación física detectada durante las 
pruebas fue la justificación técnica principal para diseñar una arquitectura 
de control desacoplada basada en repulsión láser (LiDAR) en el script final 
'codigo_final.py'.
=============================================================================
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import math
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

class OdometryNav:
    def __init__(self):
        rospy.init_node('odom_nav_node', anonymous=True)
        
        self.cmd_pub = rospy.Publisher('/robot_1/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/robot_1/odom', Odometry, self.odom_callback)

        # Variables en el plano global imaginario
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.odom_ready = False

        # --- CONFIGURA TU PLANO Y META AQUÍ ---
        # 0=Frente(+X), 90=Izquierda(+Y), 180=Atras(-X), 270=Derecha(-Y)
        self.start_yaw_deg = 270
        self.yaw_offset = math.radians(self.start_yaw_deg)

        self.target_x = 1.0  # Coordenada X global
        self.target_y = 1.5  # Coordenada Y global

        # Tolerancias
        self.tol_dist = 0.05
        self.tol_angle = 0.05

        self.state = "GIRANDO"

    def euler_yaw_from_quaternion(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def odom_callback(self, msg):
        # 1. Extraemos los datos crudos del hardware de ROS (donde arranco = 0)
        raw_x = msg.pose.pose.position.x
        raw_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        raw_yaw = self.euler_yaw_from_quaternion(q)

        # 2. TRANSFORMACIÓN MATEMÁTICA AL PLANO GLOBAL DEL USUARIO
        # Aplicamos matriz de rotación 2D para girar el universo entero
        self.x = raw_x * math.cos(self.yaw_offset) - raw_y * math.sin(self.yaw_offset)
        self.y = raw_x * math.sin(self.yaw_offset) + raw_y * math.cos(self.yaw_offset)
        
        # Sumamos el offset al ángulo y lo normalizamos
        self.yaw = self.normalize_angle(raw_yaw + self.yaw_offset)
        
        self.odom_ready = True

    def normalize_angle(self, angle):
        while angle > math.pi: angle -= 2.0 * math.pi
        while angle < -math.pi: angle += 2.0 * math.pi
        return angle

    def loop(self):
        rate = rospy.Rate(15)
        rospy.loginfo("Esperando odometria...")
        
        while not rospy.is_shutdown() and not self.odom_ready:
            rate.sleep()

        rospy.loginfo(f"¡Inicio OK! El robot se asume en X:0, Y:0 apuntando a {self.start_yaw_deg} grados.")
        rospy.loginfo(f"Navegando hacia X:{self.target_x} , Y:{self.target_y}")

        while not rospy.is_shutdown():
            dx = self.target_x - self.x
            dy = self.target_y - self.y
            dist_error = math.sqrt(dx**2 + dy**2)
            
            target_angle = math.atan2(dy, dx)
            angle_error = self.normalize_angle(target_angle - self.yaw)

            twist = Twist()

            if self.state == "GIRANDO":
                if abs(angle_error) > self.tol_angle:
                    twist.angular.z = 0.8 * angle_error 
                    if twist.angular.z > 0.6: twist.angular.z = 0.6
                    if twist.angular.z < -0.6: twist.angular.z = -0.6
                    rospy.loginfo(f"GIRANDO  | Orientacion actual: {math.degrees(self.yaw):.1f} | Faltan: {math.degrees(angle_error):.1f} grados")
                else:
                    self.state = "AVANZANDO"
                    twist.angular.z = 0.0
                    rospy.loginfo("¡Alineado! Arrancando motores...")

            elif self.state == "AVANZANDO":
                if dist_error > self.tol_dist:
                    twist.linear.x = 0.4 * dist_error 
                    if twist.linear.x > 0.15: twist.linear.x = 0.15  
                    twist.angular.z = 0.6 * angle_error
                    rospy.loginfo(f"MARCHANDO | Pos global: X={self.x:.2f} Y={self.y:.2f} | Falta: {dist_error:.2f} m")
                else:
                    self.state = "LLEGAMOS"
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0

            elif self.state == "LLEGAMOS":
                rospy.loginfo("🎯 ¡META ALCANZADA CON ÉXITO!")
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.cmd_pub.publish(twist)
                break

            self.cmd_pub.publish(twist)
            rate.sleep()

if __name__ == '__main__':
    try:
        nav = OdometryNav()
        nav.loop()
    except rospy.ROSInterruptException:
        pass
