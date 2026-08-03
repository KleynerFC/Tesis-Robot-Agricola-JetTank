"""
=============================================================================
PROYECTO: Robot Móvil Terrestre para Evaluación de Cultivos (Tesis UPSE)
ARCHIVO: test_servo_camara.py
AUTOR: Kleyner Fabricio Flores Cedeño
=============================================================================

DESCRIPCIÓN:
Script de validación de hardware para probar el servomotor de la cámara PTZ 
mediante el teclado de la computadora (vía SSH).

IMPORTANCIA PARA LA TESIS:
Este script fue fundamental para descubrir una limitación física de la 
plataforma Hiwonder JetTank. Al realizar las pruebas de control, se evidenció 
que el mecanismo PTZ de la cámara solo posee servomotor electrónico en su 
eje Pan (horizontal). El eje Tilt (vertical) no responde a comandos de ROS 
y debe ajustarse manualmente por fricción mecánica antes de iniciar la misión.
Gracias a esta validación, en el script final 'solo_navegacion.py' se 
programó el movimiento automático de la cámara solo en el eje Pan (45° a la 
derecha y 45° a la izquierda) para inspeccionar los lados del surco.

USO:
Ejecutar en terminal SSH. Usar teclas:
[A] Mirar a la Izquierda (-90 grados)
[D] Mirar a la Derecha (90 grados)
[W] Mirar al Frente (Centro: 0 grados)
[Q] o [Ctrl+C] Salir y centrar la cámara de forma segura.
=============================================================================
"""

#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64
import sys
import select
import termios
import tty

class ProbadorServos:
    def __init__(self):
        rospy.init_node('test_servo_camara')
        
        # Topico confirmado en hardware para el eje Pan (Horizontal)
        self.pan_pub = rospy.Publisher('/joint1_controller/command', Float64, queue_size=1)
        
        # Nota: El topico del eje Tilt (/joint2_controller/command) se omite en 
        # este bucle de control porque se comprobó fisicamente que el robot 
        # no tiene servomotor vertical, es ajuste manual por friccion.
        
        rospy.sleep(1) 
        rospy.loginfo("Control de Servo Pan Inicializado.")
        
    def mover_servo(self, pan_angulo):
        pan_rad = pan_angulo * (3.14159 / 180.0)
        self.pan_pub.publish(Float64(pan_rad))
        rospy.loginfo(f"Camara apuntando a -> PAN: {pan_angulo} grados")

    def leer_tecla(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            r, _, _ = select.select([sys.stdin], [], [], 0.1)
            if r:
                ch = sys.stdin.read(1)
            else:
                ch = ''
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    def bucle_control(self):
        print("\n--- PANEL DE CONTROL DE CAMARA ---")
        print("Usa las teclas para apuntar a las hileras:")
        print("[A] Mirar a la Izquierda (-90 grados)")
        print("[D] Mirar a la Derecha (90 grados)")
        print("[W] Mirar al Frente (Centro)")
        print("[Q] o [Ctrl+C] Salir")
        print("----------------------------------")

        pan = 0.0
        self.mover_servo(pan)

        while not rospy.is_shutdown():
            tecla = self.leer_tecla().lower()
            
            # Reconoce la 'q' o el Ctrl+C (\x03)
            if tecla == 'q' or tecla == '\x03':
                print("\nSaliendo y centrando camara de forma segura...")
                self.mover_servo(0.0)
                rospy.sleep(0.5)
                break
            
            # Si 'A' mueve a la derecha en tu robot, invierte el signo aqui
            elif tecla == 'a':
                pan = -90.0  
                self.mover_servo(pan)
            elif tecla == 'd':
                pan = 90.0   
                self.mover_servo(pan)
            elif tecla == 'w':
                pan = 0.0   
                self.mover_servo(pan)

if __name__ == '__main__':
    try:
        probador = ProbadorServos()
        probador.bucle_control()
    except rospy.ROSInterruptException:
        pass
