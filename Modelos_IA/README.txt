=============================================================================
PROYECTO: Robot Móvil Terrestre para Evaluación de Cultivos (Tesis UPSE)
CARPETA: Modelos_IA
=============================================================================

Esta carpeta contiene los tres estados del modelo de Red Neuronal 
Convolucional (YOLOv8 Nano) utilizado para la detección y clasificación 
fitosanitaria (Sana, Plaga, Estrés) en el robot.

El modelo evolucionó siguiendo un flujo de optimización industrial 
(Edge Computing) para lograr ejecutarse en tiempo real en la GPU de la 
Jetson Nano. Los archivos son los siguientes:

1. best.pt
   - Descripción: Pesos nativos del framework PyTorch.
   - Origen: Resultado directo del entrenamiento por Transfer Learning en Google Colab (GPU Tesla T4).
   - Uso: Es el formato estándar para hacer inferencia en PC o modificar la arquitectura.

2. best.onnx
   - Descripción: Grafo computacional agnóstico (Open Neural Network Exchange).
   - Origen: Exportación del archivo .pt para estandarizar los operadores lógicos.
   - Uso: Formato puente necesario para la compilación en TensorRT.

3. best_engine.rar (dentro está el best.engine)
   - Descripción: Archivo binario compilado para TensorRT v8.2.1.
   - Origen: Compilado directamente en la NVIDIA Jetson Nano usando trtexec.
   - Uso: Es el archivo final que ejecuta el robot en campo. Al estar compilado 
     a bajo nivel, reduce la latencia a 38.95 ms (~25.7 FPS).

---------------------------------------------------------------------------
⚠️ ADVERTENCIA TÉCNICA IMPORTANTE SOBRE EL ARCHIVO .ENGINE ⚠️
---------------------------------------------------------------------------
El archivo .engine NO es multiplataforma. Al estar compilado con TensorRT, 
queda vinculado físicamente a la arquitectura de la GPU específica de la 
máquina donde se compiló.

Este .engine fue compilado específicamente para la GPU Maxwell de 128 núcleos 
de la NVIDIA Jetson Nano. 

- Si intentas ejecutar este .engine en una PC de escritorio o laptop, 
  arrojará un error fatal de incompatible.
- Si deseas probar el modelo en otra PC, utiliza el archivo best.pt.
- Para usar el .engine en otro robot JetTank, usando el mismo código principal, descomprime el archivo .rar 
  y coloca best.engine en la siguiente ruta del robot:
  /home/hiwonder/jettank_ws/src/src/jettank_control/jettank_control/
=============================================================================
