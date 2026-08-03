## Tesis-Robot-Agricola-JetTank
Implementación de Robot Móvil Terrestre para Evaluación de Cultivos  
Trabajo de Integración Curricular presentado para la obtención del título de Ingeniero en Electrónica y Automatización en la Universidad Estatal Península de Santa Elena (UPSE).

Autor: Kleyner Fabricio Flores Cedeño  
Tutor: Ing. Junior Figueroa, M.Sc.  
Año: 2026  

# Descripción del Proyecto

Sistema robótico basado en la plataforma Hiwonder JetTank ROS y NVIDIA Jetson Nano. Utiliza una arquitectura de control desacoplada: navegación reactiva mediante LiDAR e IMU, y percepción visual mediante YOLOv8 optimizado con TensorRT para la clasificación fitosanitaria en tiempo real.

# Estructura del Repositorio

El repositorio contiene no solo el código de operación final, sino también los scripts de prueba, modelos 3D y pesos de la red neuronal utilizados durante el desarrollo y validación del hardware:

* /Codigo_Principal: Contiene el script codigo_final.py. Controla la navegación por LiDAR, giros en U direccionales, conteo de plantas multiclase por enfriamiento temporal (cooldown) y la Interfaz Gráfica (GUI) en Tkinter.
* /Mapeo_Teleoperado: Contiene el script mapeo_manual.py para mapeo SLAM mediante joystick, usado para validar el funcionamiento de los encoders a baja velocidad.
* /Scripts_de_Validacion: Contiene códigos auxiliares empleados para el diagnóstico de sensores y redes:
    - odom.py: Validación de odometría pura y cinemática (Dead Reckoning).
    - prueba_ia.py: Validación offline de la inferencia de TensorRT en imágenes estáticas.
    - video_ia.py y video_ia_grabar.py: Pruebas de estrés de streaming y grabación de video procesado.
    - tomar_foto.py: Script de telemetría asíncrona por SFTP para eludir la saturación de red.
    - test_servo_camara.py: Diagnóstico de los servomotores PTZ de la cámara.
* /Lanzadores_Despliegue: Contiene los archivos .desktop (íconos de pantalla táctil) utilizados para el arranque autónomo del SLAM y el script principal en el robot, sin dependencia de redes Wi-Fi.
* /Modelos_3D: Contiene el archivo de la biomasa sintética (lechugas impresas en 3D) utilizada para el entrenamiento y validación tras la depredación del cultivo real.
* /Modelos_IA: Contiene los 3 estados del modelo YOLOv8 Nano: best.pt (PyTorch), best.onnx (ONNX) y best.engine (compilado para Jetson Nano, comprimido en .rar). Incluye un archivo de texto con las advertencias de compatibilidad de TensorRT.

# Hardware Requerido

* Robot Hiwonder JetTank (Advanced Kit)
* NVIDIA Jetson Nano (4GB)
* Sensor LiDAR EAI YDLIDAR G4
* Cámara RGB-D Astra Pro
* Dongle USB para Joystick (para mapeo manual)
* Batería LiPo 11.1V 6000mAh
