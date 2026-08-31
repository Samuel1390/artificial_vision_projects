import mediapipe as mp
import cv2 as cv
import os
import time
import numpy as np
from draw_hand import draw_hand
from pathlib import Path
from is_hand_closed import is_hand_closed

class StrokeStabilizer:
    """
    Implementa la media movil exponencial (EMA), con el objetivo de reducir el ruido en los movimientos de la mano.
    """
    def __init__(self, smoothing_factor=0.25):
        self.alpha = smoothing_factor
        self.stabilized_x = None
        self.stabilized_y = None

    def stabilize(self, target_x, target_y):
        if self.stabilized_x is None or self.stabilized_y is None:
            self.stabilized_x = target_x
            self.stabilized_y = target_y
        else:
            self.stabilized_x = int(self.alpha * target_x + (1 - self.alpha) * self.stabilized_x)
            self.stabilized_y = int(self.alpha * target_y + (1 - self.alpha) * self.stabilized_y)
        return self.stabilized_x, self.stabilized_y

    def reset(self):
        self.stabilized_x = None
        self.stabilized_y = None


root_path = Path(__file__).parent.parent
hand_model_path = str(root_path / "models/hand_landmarker.task")

"""
Referencias las clases internas de la API de arquitectura de tareas de MediaPipe.
HandLandmarker maneja la inicialización del modelo y el procesamiento de la canalización.
"""
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

def hand_callback(result, output_image, timestamp_ms):
    """
    Esta función se ejecuta de manera asíncrona en un hilo de fondo separado
    cada vez que MediaPipe termina de procesar un fotograma. Actualiza el estado global
    con las estructuras de seguimiento de puntos de referencia de la mano detectados.
    """
    global hand_landmarks_list
    if result.hand_landmarks:
        hand_landmarks_list = result.hand_landmarks
    else:
        hand_landmarks_list = None

"""
Configura MediaPipe para una aplicación de cámara de video continua.
El modo LIVE_STREAM le dice al framework que ejecute bucles de procesamiento no bloqueantes,
eliminando tareas de seguimiento si el hardware se queda atrás de la alimentación real de la cámara.
"""
hand_landmarker_options = HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=hand_model_path),
    running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
    num_hands=1,
    result_callback=hand_callback
)
hand_landmarks_list = None

"""
Inicializa la captura de video, conectada a la webcam local predeterminada (índice 0).
"""
camera = cv.VideoCapture(0)

canvas = None
prev_x, prev_y = None, None
brush_color = (255, 255, 0)
brush_thickness = 5

stabilizer = StrokeStabilizer(smoothing_factor=0.25)

with HandLandmarker.create_from_options(hand_landmarker_options) as landmarker:
    while True:
        """
        Obtiene un solo fotograma de la canalización del dispositivo. 'ret' es un booleano que rastrea el éxito,
        y 'frame' es una matriz NumPy multidimensional estructurada.
        """
        ret, frame = camera.read()
        frame = cv.flip(frame, 1)
        if not ret:
            break
        
        h, w, _ = frame.shape

        if canvas is None:
            """
            np.zeros_like crea una matriz de lienzo de dibujo en blanco, oscuro, que coincide con
            las dimensiones exactas de los datos y los tipos de datos (uint8) del fotograma de video fuente.
            """
            canvas = np.zeros_like(frame)

        """
        Los flujos de seguimiento asíncronos de MediaPipe requieren estrictamente marcas de tiempo enteras de milisegundos crecientes monótonamente 
        para alinear los pasos de procesamiento.
        """
        timestamp_ms = int(time.time() * 1000)
        """
        Los fotogramas de OpenCV se leen como matrices BGR, pero los pesos del modelo MediaPipe esperan matrices RGB.
        cv.cvtColor transforma el diseño de la matriz de representación de color antes del procesamiento.
        """
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        """
        Envuelve la matriz NumPy cruda en un bloque de objetos MediaPipe Image gestionado por memoria
        usando el formato SRGB sin comprimir estándar.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        """
        Envía el fotograma al hilo del oleoducto de fondo de la red neuronal sin bloqueo.
        Los resultados llegarán a través de la función 'hand_callback' registrada previamente.
        """
        landmarker.detect_async(mp_image, timestamp_ms)

        if hand_landmarks_list:
            for hand_landmarks in hand_landmarks_list:
                raw_x = int(hand_landmarks[8].x * w)
                raw_y = int(hand_landmarks[8].y * h)

                """
                Evalúa qué dedos específicos están actualmente extendidos. Debido a que la coordenada 0
                está en la parte superior de la vista, una coordenada de punta con un .y más bajo que su articulación interior
                prueba que está físicamente extendida hacia arriba.
                """
                fingers_up = [
                    hand_landmarks[8].y < hand_landmarks[6].y,
                    hand_landmarks[12].y < hand_landmarks[10].y,
                    hand_landmarks[16].y < hand_landmarks[14].y,
                    hand_landmarks[20].y < hand_landmarks[18].y
                ]

                if fingers_up[0] and not any(fingers_up[1:]):
                    ix, iy = stabilizer.stabilize(raw_x, raw_y)
                    cv.circle(frame, (ix, iy), brush_thickness, brush_color, -1)
                    
                    if prev_x is None or prev_y is None:
                        prev_x, prev_y = ix, iy
                    
                    cv.line(canvas, (prev_x, prev_y), (ix, iy), brush_color, brush_thickness)
                    prev_x, prev_y = ix, iy

                elif fingers_up[0] and fingers_up[1] and not any(fingers_up[2:]):
                    ix, iy = stabilizer.stabilize(raw_x, raw_y)
                    cv.circle(frame, (ix, iy), 5, (255, 255, 255), 1)
                    
                    if prev_x is None or prev_y is None:
                        prev_x, prev_y = ix, iy
                    """
                    Un trazo de borrador se logra dibujando pistas de línea en la matriz de lienzo
                    usando negro puro (0, 0, 0), que coincide con el estado de color de fondo del lienzo.
                    """
                    cv.line(canvas, (prev_x, prev_y), (ix, iy), (0, 0, 0), 10)
                    prev_x, prev_y = ix, iy

                elif fingers_up[0] and fingers_up[1] and fingers_up[2] and not fingers_up[3]:
                    ix, iy = stabilizer.stabilize(raw_x, raw_y)
                    cv.circle(frame, (ix, iy), 15, (255, 255, 255), 1)
                    
                    if prev_x is None or prev_y is None:
                        prev_x, prev_y = ix, iy
                    cv.line(canvas, (prev_x, prev_y), (ix, iy), (0, 0, 0), 30)
                    prev_x, prev_y = ix, iy

                elif is_hand_closed(hand_landmarks):
                    prev_x, prev_y = None, None
                    stabilizer.reset()
                    """
                    Limpia los dibujos activos instantáneamente restableciendo todos los píxeles del lienzo de vuelta a 0 (Negro).
                    """
                    canvas = np.zeros_like(frame)
                    
                    """
                    Superpone texto dinámico en el fotograma de la vista de la cámara. Acepta estilos de fuente, 
                    scale metrics, colors (BGR Red), and border stroke weights.
                    """
                    cv.putText(frame, "CANVAS CLEARED", (50, 50), 
                               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                else:
                    prev_x, prev_y = None, None
                    stabilizer.reset()
                    if all(fingers_up):
                        cv.circle(frame, (raw_x, raw_y), 15, (0, 0, 255), 2)

                draw_hand(frame, hand_landmarks, skeleton_width=1, point_radius=3)
        else:
            prev_x, prev_y = None, None
            stabilizer.reset()

        """
        cv.convertScaleAbs cambia el brillo de la imagen. Multiplicar cada índice de matriz 
        elemento por alpha=0.8 oscurece el video de fondo crudo para que las líneas resalten.
        """
        darker_frame = cv.convertScaleAbs(frame, alpha=0.8, beta=0)

        """
        Convierte el canal de la imagen del lienzo del espacio de matriz de color de 3 canales a un diseño de escala de grises de 1 canal.
        """
        img_gray = cv.cvtColor(canvas, cv.COLOR_BGR2GRAY)
        """
        Aplica una operación de umbral binario. Cualquier píxel del lienzo mayor que 20 se convierte a 0, 
        y todo lo demás se convierte en 255 (Blanco). Esto aísla las líneas dibujadas en una máscara de plantilla invertida.
        """
        _, img_inv = cv.threshold(img_gray, 20, 255, cv.THRESH_BINARY_INV)
        """
        Convierte la máscara binaria de un solo canal de nuevo a 3 canales para permitir cálculos de matriz bit a bit.
        """
        img_inv = cv.cvtColor(img_inv, cv.COLOR_GRAY2BGR)
        
        """
        Realiza una operación de matriz AND lógica. Quema cortes de silueta oscura dentro del fotograma de video
        precisamente donde residen los trazos del pincel en el lienzo.
        """
        darker_frame = cv.bitwise_and(darker_frame, img_inv)
        """
        Combina los arrays algebraicamente mediante adición directa de matriz. El color del trazo del pincel llena 
        los huecos de la silueta perfectamente sin artefactos de mezcla.
        """
        final_frame = cv.add(darker_frame, canvas)

        """
        Crea una ventana de vista de contenedor de renderizado de UI de alto nivel que muestra la matriz final fusionada.
        """
        cv.imshow("Virtual Canvas", final_frame)
        
        """
        cv.waitKey(1) retrasa la ejecución del hilo durante 1 ms esperando las señales de sondeo del teclado del hardware.
        Enmascarar con 0xFF aísla los 8 bits inferiores que representan coincidencias de caracteres ASCII estándar.
        """
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            canvas = np.zeros_like(frame)

camera.release()
cv.destroyAllWindows()