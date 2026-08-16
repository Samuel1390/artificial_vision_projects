import cv2
import numpy as np
import mediapipe as mp
import math
from pathlib import Path
import time

class HandFilterApp:
    def __init__(self, model_path=None):
        # Configuración inicial de rutas y variables de MediaPipe
        if model_path is None:
            self.model_path = Path(__file__).parent.parent / "models" / "hand_landmarker.task"
        else:
            self.model_path = Path(model_path)
            
        self.BaseOptions = mp.tasks.BaseOptions
        self.HandLandmarker = mp.tasks.vision.HandLandmarker
        self.HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        
        self.current_filter = 0
        self.is_hidden = False
        self.landmarker = None
        
        # Generación de las matrices de texturas en memoria durante la inicialización
        self.textures = [
            self._create_diagonal_texture(),
            self._create_hexagonal_texture(),
            self._create_triangular_texture()
        ]
        
        # Obtenemos las dimensiones de la textura generada (alto y ancho)
        self.tex_h, self.tex_w = self.textures[0].shape[:2]
        
        # Definimos los 4 vértices origen (src_pts) de nuestra textura plana 2D.
        # Orden: Arriba-Izquierda, Arriba-Derecha, Abajo-Derecha, Abajo-Izquierda.
        # Esto es crucial para calcular la matriz de homografía más adelante.
        self.src_pts = np.float32([[0, 0], [self.tex_w, 0], [self.tex_w, self.tex_h], [0, self.tex_h]])

    def _init_landmarker(self):
        # Inicializa el modelo en modo VIDEO para procesamiento síncrono (bloqueante frame a frame)
        options = self.HandLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=self.VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.landmarker = self.HandLandmarker.create_from_options(options)

    # ==========================================
    # GENERADORES DE TEXTURAS
    # ==========================================
    
    def _create_diagonal_texture(self, w=400, h=400):
        # Crea un tensor 3D de ceros (imagen negra) de dimensiones h x w x 3 canales (RGB)
        tex = np.zeros((h, w, 3), dtype=np.uint8)
        # Itera desde -h hasta w*2 para asegurar que las líneas diagonales cubran toda el área
        for i in range(-h, w * 2, 20):
            # Dibuja líneas desde la coordenada (i, 0) hasta (i+h, h). 
            # Como X incrementa al mismo ritmo que Y, la pendiente es de 45 grados.
            cv2.line(tex, (i, 0), (i + h, h), (255, 255, 255), 4)
        return tex

    def _create_hexagonal_texture(self, w=400, h=400):
        # Crea el lienzo negro en formato BGR (uint8)
        tex = np.zeros((h, w, 3), dtype=np.uint8)
        # 's' representa la longitud del lado del hexágono
        s = 14
        # Bucle anidado para construir una cuadrícula (grid) hexagonal.
        # La distancia vertical entre centros es 1.5 * lado.
        for row in range(-h, h * 2, int(s * 1.5)):
            # La distancia horizontal entre centros es lado * raíz(3).
            for col in range(-w, w * 2, int(s * np.sqrt(3))):
                # Calcula un offset horizontal alterno (0 o 1) dependiendo de la fila par/impar
                # Esto es lo que intercala los hexágonos creando el patrón de panal.
                offset = (row // int(s * 1.5)) % 2
                x = col + offset * int(s * np.sqrt(3) / 2)
                y = row
                
                pts = []
                # Calcula los 6 vértices del hexágono usando trigonometría básica (coordenadas polares a cartesianas)
                for i in range(6):
                    # El ángulo avanza en pasos de 60 grados (pi/3 radianes), con un desfase de 30 grados para orientarlos.
                    angle = np.pi / 180 * (60 * i + 30)
                    pts.append([int(x + s * np.cos(angle)), int(y + s * np.sin(angle))])
                # Dibuja el polígono cerrado basado en los 6 puntos calculados.
                cv2.polylines(tex, [np.array(pts)], True, (255, 255, 255), 2)
        return tex

    def _create_triangular_texture(self, w=400, h=400):
        tex = np.zeros((h, w, 3), dtype=np.uint8)
        step = 40
        # Crea una red de triángulos superponiendo líneas diagonales positivas, negativas y horizontales.
        for i in range(-h, w * 2, step):
            cv2.line(tex, (i, 0), (i + h, h), (255, 255, 255), 3)       # Diagonal \
            cv2.line(tex, (i, h), (i + h, 0), (255, 255, 255), 3)       # Diagonal /
            cv2.line(tex, (0, i), (w, i), (255, 255, 255), 3)           # Horizontal -
        return tex

    # ==========================================
    # LÓGICA CORE DE FILTROS Y RENDERIZADO
    # ==========================================

    def _apply_color_filter(self, roi, filter_idx):
        # convertScaleAbs realiza la operación: O(x,y) = |alpha * I(x,y) + beta|
        # Esto aumenta el contraste (alpha=1.4) y el brillo (beta=40) del fragmento (Region of Interest)
        filtered_roi = cv2.convertScaleAbs(roi, alpha=1.4, beta=40)
        
        # Separa los canales de color en matrices 2D individuales (Blue, Green, Red)
        b, g, r = cv2.split(filtered_roi)
        
        # Se manipulan los canales casteándolos a int16 temporalmente para evitar el desbordamiento (overflow) 
        # característico del uint8 (donde 255 + 1 = 0).
        if filter_idx == 0:   # Filtro Azul (Cyan)
            # Aumentamos agresivamente el canal azul y reducimos el rojo.
            # np.clip restringe los valores resultantes estrictamente al rango seguro [0, 255] antes de volver a uint8.
            b = np.clip(b.astype(np.int16) + 70, 0, 255).astype(np.uint8)
            r = np.clip(r.astype(np.int16) - 30, 0, 255).astype(np.uint8)
            
        elif filter_idx == 1: # Filtro Púrpura Neón
            # Elevamos tanto el azul como el rojo (creando magenta/púrpura) y reducimos el verde.
            b = np.clip(b.astype(np.int16) + 60, 0, 255).astype(np.uint8)
            r = np.clip(r.astype(np.int16) + 60, 0, 255).astype(np.uint8)
            g = np.clip(g.astype(np.int16) - 30, 0, 255).astype(np.uint8)
            
        elif filter_idx == 2: # Filtro Amarillo
            # Elevamos el rojo y el verde (cuya mezcla aditiva produce amarillo) y hundimos el azul.
            r = np.clip(r.astype(np.int16) + 60, 0, 255).astype(np.uint8)
            g = np.clip(g.astype(np.int16) + 60, 0, 255).astype(np.uint8)
            b = np.clip(b.astype(np.int16) - 50, 0, 255).astype(np.uint8)
            
        # Volvemos a fusionar los canales matriciales en un solo tensor 3D de orden BGR.
        return cv2.merge((b, g, r))

    def _render_accordion_3d(self, frame, left_lm, right_lm, w, h):
        # Índices exactos del modelo de MediaPipe para las puntas de los 5 dedos.
        # [4: Pulgar, 8: Índice, 12: Medio, 16: Anular, 20: Meñique]
        tips_idx = [4, 8, 12, 16, 20] 
        
        # Mapea las coordenadas normalizadas (0.0 a 1.0) a píxeles absolutos (multiplicando por w y h).
        left_tips = [(int(left_lm[i].x * w), int(left_lm[i].y * h)) for i in tips_idx]
        right_tips = [(int(right_lm[i].x * w), int(right_lm[i].y * h)) for i in tips_idx]
        
        # Define los 4 sub-filtros. Cada uno se aplicará en el espacio entre dos dedos adyacentes.
        effects = [
            cv2.COLORMAP_JET,      # Segmento 0: Pulgar a Índice
            "INVERT",              # Segmento 1: Índice a Medio
            cv2.COLORMAP_BONE,     # Segmento 2: Medio a Anular (Se le añadirá el patrón de líneas)
            cv2.COLORMAP_MAGMA     # Segmento 3: Anular a Meñique (Se le añadirá el patrón hexagonal)
        ]
        
        # Iteramos 4 veces para construir y renderizar los 4 cuadriláteros contiguos.
        for i in range(4):
            # Definimos el polígono de 4 vértices conectando los dedos homólogos izquierdo y derecho.
            # El orden es cíclico: Arriba-Izq, Arriba-Der, Abajo-Der, Abajo-Izq.
            quad = np.array([
                left_tips[i],       
                right_tips[i],      
                right_tips[i+1],    
                left_tips[i+1]      
            ], dtype=np.int32)
            
            # Calculamos el "Bounding Box" (rectángulo delimitador) ortogonal que contiene al cuadrilátero.
            # np.min y np.max extraen los extremos absolutos en el eje X (columna 0) y Y (columna 1).
            xmin, xmax = np.min(quad[:, 0]), np.max(quad[:, 0])
            ymin, ymax = np.min(quad[:, 1]), np.max(quad[:, 1])
            
            # Recortamos las coordenadas para asegurarnos de que no excedan los límites reales de la pantalla.
            xmin, xmax, ymin, ymax = max(0, xmin), min(w, xmax), max(0, ymin), min(h, ymax)
            
            # Calculamos las dimensiones del rectángulo delimitador.
            roi_w, roi_h = xmax - xmin, ymax - ymin
            if roi_w <= 0 or roi_h <= 0:
                continue # Si el área colapsa o es negativa, saltamos esta iteración para evitar crashes.
                
            # Extraemos la sub-matriz de la imagen original (el Bounding Box exacto).
            roi = frame[ymin:ymax, xmin:xmax]
            
            # Trasladamos las coordenadas del cuadrilátero global al sistema de coordenadas local del ROI.
            # Restamos [xmin, ymin] a todos los vértices.
            quad_roi = quad - np.array([xmin, ymin])
            
            # --- APLICACIÓN DE FILTRO DE COLOR BASE ---
            if effects[i] == "INVERT":
                # bitwise_not invierte los bits (255 - valor_pixel), logrando un efecto fotográfico negativo.
                filtered_roi = cv2.bitwise_not(roi)
            else:
                # Los ColorMaps requieren una imagen de 1 solo canal (escala de grises) como entrada.
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                # applyColorMap asocia la intensidad lumínica (0-255) a una paleta de colores RGB específica.
                filtered_roi = cv2.applyColorMap(gray_roi, effects[i])
                
            # --- APLICACIÓN DE TEXTURAS (VÍA HOMOGRAFÍA) ---
            # Si estamos en el filtro metálico (índice 2) o el violeta extraño (índice 3), añadimos texturas.
            if i == 2 or i == 3:
                # Seleccionamos la textura correcta: índice 0 (diagonal) para Metálico, índice 1 (hexagonal) para Violeta.
                tex_idx = 0 if i == 2 else 1
                active_texture = self.textures[tex_idx]
                
                # cv2.getPerspectiveTransform calcula la matriz de transformación 3x3 que mapea los puntos
                # del rectángulo 2D perfecto de nuestra textura (src_pts) a los vértices irregulares del cuadrilátero (quad_roi).
                matrix = cv2.getPerspectiveTransform(self.src_pts, quad_roi.astype(np.float32))
                
                # warpPerspective aplica matemáticamente la matriz calculada para deformar la textura.
                warped_tex = cv2.warpPerspective(active_texture, matrix, (roi_w, roi_h))
                
                # addWeighted realiza una fusión alfa (Alpha Blending): Resultado = (filtro * 1.0) + (textura * 0.7)
                filtered_roi = cv2.addWeighted(filtered_roi, 1.0, warped_tex, 0.7, 0)
                
            # --- ENSAMBLAJE FINAL MEDIANTE MÁSCARA BINARIA ---
            # Creamos una matriz de ceros (negro puro) del tamaño del ROI.
            mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
            # Dibujamos el cuadrilátero rellenado con blanco (255) sobre la máscara.
            cv2.fillPoly(mask, [quad_roi], 255)
            
            # Creamos una matriz booleana (True donde hay blanco, False donde hay negro).
            mask_bool = mask == 255
            # Aplicamos una indexación booleana de NumPy: Inyectamos los píxeles procesados en el ROI 
            # ÚNICAMENTE en la región donde la máscara es True (es decir, dentro del polígono exacto).
            roi[mask_bool] = filtered_roi[mask_bool]
            
            # Dibujamos el contorno blanco en 3D para definir la separación entre los filtros del acordeón.
            cv2.polylines(frame, [quad], isClosed=True, color=(255, 255, 255), thickness=2)

    def _render_standard_filter(self, frame, dst_pts, w, h):
        # Misma lógica fundamental de Bounding Box, transformación de perspectiva y enmascaramiento 
        # detallada en el método de renderizado del acordeón, pero aplicado a un solo cuadrilátero general.
        xmin, xmax = int(np.floor(np.min(dst_pts[:, 0]))), int(np.ceil(np.max(dst_pts[:, 0])))
        ymin, ymax = int(np.floor(np.min(dst_pts[:, 1]))), int(np.ceil(np.max(dst_pts[:, 1])))
        xmin, xmax, ymin, ymax = max(0, xmin), min(w, xmax), max(0, ymin), min(h, ymax)

        roi_w, roi_h = xmax - xmin, ymax - ymin
        if roi_w > 0 and roi_h > 0:
            dst_pts_roi = dst_pts - np.array([xmin, ymin], dtype=np.float32)
            
            mask_roi = np.zeros((roi_h, roi_w), dtype=np.uint8)
            cv2.fillPoly(mask_roi, [dst_pts_roi.astype(np.int32)], 255)
            roi_frame = frame[ymin:ymax, xmin:xmax]

            active_texture = self.textures[self.current_filter]
            matrix = cv2.getPerspectiveTransform(self.src_pts, dst_pts_roi)
            warped_lines = cv2.warpPerspective(active_texture, matrix, (roi_w, roi_h))
            
            filtered_roi = self._apply_color_filter(roi_frame, self.current_filter)
            blended_roi = cv2.addWeighted(filtered_roi, 1.0, warped_lines, 0.7, 0)

            mask_boolean = mask_roi == 255
            roi_frame[mask_boolean] = blended_roi[mask_boolean]

            cv2.polylines(frame, [dst_pts.astype(np.int32)], isClosed=True, color=(255, 255, 255), thickness=2)

    def _process_frame(self, frame, timestamp_ms):
        # Convertimos la imagen de BGR (formato nativo de OpenCV) a RGB (requerido por MediaPipe).
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Ejecución síncrona de la inferencia de redes neuronales de MediaPipe.
        results = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        
        hands_data = []
        if results and results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                # Calculamos el centro de masa de la mano (promediando todos sus landmarks)
                # para poder determinar matemáticamente cuál mano está a la izquierda y cuál a la derecha.
                cx = np.mean([lm.x for lm in hand_landmarks])
                cy = np.mean([lm.y for lm in hand_landmarks])
                
                hands_data.append({
                    'landmarks': hand_landmarks,
                    'cx': cx * w,
                    'cy': cy * h
                })

        if len(hands_data) >= 2:
            # Ordenamos el arreglo de diccionarios en base al eje X ('cx').
            # Así aseguramos que hands_data[0] es la mano izquierda y hands_data[-1] la derecha.
            hands_data = sorted(hands_data, key=lambda d: d['cx'])
            left_hand = hands_data[0]
            right_hand = hands_data[-1]

            # Establecemos el umbral dinámico basado en el 8% del ancho de la pantalla actual.
            dist_threshold = w * 0.08
            # Aplicamos la fórmula de distancia euclidiana (Teorema de Pitágoras) para calcular la cercanía.
            hands_dist = math.hypot(left_hand['cx'] - right_hand['cx'], left_hand['cy'] - right_hand['cy'])

            # Lógica de Máquina de Estados Finita (FSM) para el cambio de filtros.
            if hands_dist < dist_threshold:
                if not self.is_hidden:
                    # El módulo (%) asegura que el contador reinicie a 0 tras llegar a 3 (0, 1, 2, 3).
                    self.current_filter = (self.current_filter + 1) % 4
                    self.is_hidden = True
            else:
                self.is_hidden = False

            if not self.is_hidden:
                if self.current_filter < 3:
                    # Renderiza el cuadrilátero básico entre pulgares e índices
                    thumb_L = [int(left_hand['landmarks'][4].x * w), int(left_hand['landmarks'][4].y * h)]
                    index_L = [int(left_hand['landmarks'][8].x * w), int(left_hand['landmarks'][8].y * h)]
                    thumb_R = [int(right_hand['landmarks'][4].x * w), int(right_hand['landmarks'][4].y * h)]
                    index_R = [int(right_hand['landmarks'][8].x * w), int(right_hand['landmarks'][8].y * h)]
                    
                    dst_pts = np.float32([index_L, index_R, thumb_R, thumb_L])
                    self._render_standard_filter(frame, dst_pts, w, h)
                else:
                    # Despliega la malla compleja del cuarto estado
                    self._render_accordion_3d(frame, left_hand['landmarks'], right_hand['landmarks'], w, h)

        return frame

    def live_stream(self):
        # Función orquestadora para captura y renderizado vía hardware local (Webcam).
        self._init_landmarker()
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        start_time = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Efecto espejo en el eje horizontal (eje Y = 1) para naturalidad visual.
            frame = cv2.flip(frame, 1)
            # El timestamp garantiza que el motor interno de tracking asocie correctamente los frames consecutivos.
            timestamp_ms = int((time.time() - start_time) * 1000)
            
            processed_frame = self._process_frame(frame, timestamp_ms)
            
            cv2.imshow("Hand Filter App - Live Stream", processed_frame)
            # Termina el bucle si detecta el ingreso de la tecla ASCII correspondiente a 'q'.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def process_video(self, input_path, output_path):
        # Orquestador optimizado para lectura y escritura de buffers de video en almacenamiento.
        self._init_landmarker()
        cap = cv2.VideoCapture(input_path)
        
        # Extrae metadata del archivo origen para configurar el constructor (VideoWriter) del archivo destino.
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # fourcc (Four-Character Code) es el identificador del códec de compresión que se utilizará.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Generación determinista del tiempo basada en los FPS reales del archivo.
            timestamp_ms = int((frame_idx / fps) * 1000)
            frame_idx += 1
            
            processed_frame = self._process_frame(frame, timestamp_ms)
            out.write(processed_frame)
            
            cv2.imshow("Processing Video...", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Video exportado con éxito a: {output_path}")

if __name__ == "__main__":
    app = HandFilterApp()
    app.live_stream() # aqui puedes grabar un video en tiempo real, requiere webcam o tambien puedes usar el celular
    # o tambien puedes usar un video pregrabado
    # desktop_dir = Path().home() / 'Desktop'
    # app.process_video(str(desktop_dir / 'input-reel.mp4'), str(desktop_dir / 'output_dir.mp4'))