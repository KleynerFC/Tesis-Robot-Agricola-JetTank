# Tesis-Robot-Agricola-JetTank
# Implementación de Robot Móvil Terrestre para Evaluación de Cultivos

# Trabajo de Integración Curricular presentado para la obtención del título de Ingeniero en Electrónica y Automatización en la Universidad Estatal Península de Santa Elena (UPSE).

Autor: Kleyner Fabricio Flores Cedeño
Tutor: Ing. Junior Figueroa, M.Sc.
Año: 2026

Descripción del Proyecto
Sistema robótico basado en la plataforma Hiwonder JetTank ROS y NVIDIA Jetson Nano. Utiliza una arquitectura de control desacoplada: navegación reactiva mediante LiDAR e IMU, y percepción visual mediante YOLOv8 optimizado con TensorRT.

Estructura del Repositorio
El repositorio contiene no solo el código de operación final, sino también los scripts de prueba utilizados durante la fase de desarrollo y validación de hardware:

/Codigo_Principal: Contiene el script solo_navegacion.py. Controla la navegación por LiDAR, giros en U, conteo de plantas por enfriamiento temporal (cooldown) y la Interfaz Gráfica (GUI) en Tkinter.
/Mapeo_Teleoperado: Contiene el script mapeo_manual.py para mapeo SLAM mediante joystick, usado para validar el funcionamiento de los encoders.
/Scripts_de_Validacion: Contiene códigos auxiliares empleados para el diagnóstico de sensores (lectura de encoders, telemetría de imágenes por SFTP, y calibración de cámara).
Hardware Requerido
Robot Hiwonder JetTank (Advanced Kit)
NVIDIA Jetson Nano (4GB)
Sensor LiDAR EAI YDLIDAR G4
Cámara RGB-D Astra Pro
