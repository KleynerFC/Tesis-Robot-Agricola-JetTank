"""
=============================================================================
PROYECTO: Robot Móvil Terrestre para Evaluación de Cultivos (Tesis UPSE)
ARCHIVO: tomar_foto.py
AUTOR: Kleyner Fabricio Flores Cedeño
=============================================================================

DESCRIPCIÓN:
Script de telemetría asíncrona diseñado para extraer capturas fotográficas 
estáticas desde la cámara del robot hacia la estación base mediante SFTP.

IMPORTANCIA PARA LA TESIS:
Este script resolvió dos problemas críticos documentados en la tesis:
1. El cuello de botella de red (Saturación X11 por SSH): Al tomar una sola 
   foto bajo demanda en lugar de transmitir video continuo, se redujo el 
   consumo de ancho de banda de 15 Mbps a menos de 2 Mbps.
2. La incompatibilidad de cv_bridge en ROS Melodic con Python 3: El script 
   implementa un "bypass" que lee los bytes crudos del mensaje de ROS usando 
   numpy.frombuffer y corrige manualmente la inversión de espacio de color 
   (de RGB a BGR) para que OpenCV pueda procesarlo y guardarlo en disco.

FUNCIONAMIENTO:
El nodo se suscribe al tópico de imagen, captura el primer frame disponible, 
lo guarda en el disco de la Jetson Nano como 'vista_robot.jpg' y cierra el 
nodo limpiamente para liberar la CPU.
=============================================================================
"""

#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image

def callback_imagen(msg):
    try:
        # 1. Decodificar los bytes crudos
        # Asumimos que la imagen viene en 3 canales (RGB/BGR)
        img_raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        
        # 2. Solución al bypass de cv_bridge: 
        # Convertir de RGB (como viene del driver) a BGR (como usa OpenCV)
        cv_image = cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)
        
        # 3. Guardar la imagen en la carpeta actual
        cv2.imwrite('vista_robot.jpg', cv_image)
        rospy.loginfo("Foto guardada exitosamente como 'vista_robot.jpg' con colores corregidos")
        
    except Exception as e:
        rospy.logerr(f"Error procesando la imagen: {e}")
        
    finally:
        # Apagar el programa inmediatamente despues de tomar la foto
        rospy.signal_shutdown("Mision fotografica cumplida")

def main():
    rospy.init_node('camara_espia')
    # Nos suscribimos al canal de video
    rospy.Subscriber('/astra_cam/rgb/image_raw', Image, callback_imagen)
    
    rospy.loginfo("Esperando la senal de la camara Astra Pro...")
    rospy.spin()

if __name__ == '__main__':
    main()
