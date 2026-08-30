import math
import numpy as np
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python import vision
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
import cv2

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
    pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

    for pose_landmarks in pose_landmarks_list:
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=pose_landmarks,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=pose_landmark_style,
            connection_drawing_spec=pose_connection_style)

    return annotated_image

model_path = Path(__file__).parent.parent / 'models' / 'pose_landmarker.task'
base_options = python.BaseOptions(model_asset_path=str(model_path))

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)

detector = vision.PoseLandmarker.create_from_options(options)

class PullUpCounter:

    def process_video(self, live=False, video_path=None, output_path=None):
        if live:
            cap = cv2.VideoCapture(0)
            source_name = "webcam en vivo"
        elif video_path:
            cap = cv2.VideoCapture(str(video_path))
            source_name = str(video_path)
        else:
            raise ValueError("Debe especificar `live=True` o proporcionar la ruta de un video en `video_path`.")

        if not cap.isOpened():
            print(f"Error: No se pudo abrir la fuente de video: {source_name}")
            return 0

        if output_path is None:
            if video_path:
                input_path = Path(video_path)
                out_name = f"output_{input_path.stem}.mp4"
            else:
                out_name = "output_pullups_live.mp4"
            destination_path = Path.cwd() / out_name
        else:
            destination_path = Path(output_path).resolve()

        print(f"El video procesado se guardará en: {destination_path}")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or math.isnan(fps):
            fps = 30.0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(destination_path), fourcc, fps, (frame_width, frame_height))

        stage = "down"
        pull_ups_counter = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            detection_result = detector.detect(image)
            annotated_image = draw_landmarks_on_image(rgb_frame, detection_result)
            bgr_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

            if detection_result.pose_landmarks:
                person_detected = detection_result.pose_landmarks[0]
                
                # Usamos la nariz (punto 0) como referencia para el centro de la cabeza/mentón
                # ya que MediaPipe Pose no tiene un punto específico exacto para el mentón.
                nose = person_detected[0]
                left_wrist = person_detected[15]
                right_wrist = person_detected[16]
                
                # Convertimos las coordenadas normalizadas a píxeles
                h, w, _ = frame.shape
                x1, y1 = int(left_wrist.x * w), int(left_wrist.y * h)
                x2, y2 = int(right_wrist.x * w), int(right_wrist.y * h)
                xn, yn = int(nose.x * w), int(nose.y * h)
                
                # Evitar división por cero si las manos están exactamente alineadas verticalmente
                divisor = (x2 - x1) if (x2 - x1) != 0 else 1e-6
                
                # Calculamos el punto 'y' de la barra en la posición 'x' de la cabeza
                bar_y = int(y1 + ((y2 - y1) / divisor) * (xn - x1))
                
                # Validamos si subió por encima de la barra (recordar que en OpenCV subir es restar Y)
                if yn < bar_y:
                    if stage == "down":
                        stage = "up"
                
                # Validamos si volvió a bajar (añadimos +20px de margen para evitar conteos fantasmas)
                elif yn > bar_y + 20:
                    if stage == "up":
                        stage = "down"
                        pull_ups_counter += 1
                        print(f"¡Dominada completada! Total: {pull_ups_counter}")

                # --- EFECTOS VISUALES ---
                # Dibujar la línea de la barra imaginaria entre las manos
                cv2.line(bgr_annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Dibujar un punto rojo donde la cabeza cruza la barra
                cv2.circle(bgr_annotated_image, (xn, bar_y), 6, (0, 0, 255), -1)

                # Mostrar el contador por encima del usuario (estilo Username)
                text = f"Dominadas: {pull_ups_counter}"
                
                # Calcular dinámicamente dónde centrar el texto respecto a la cabeza
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = xn - (text_size[0] // 2)
                text_y = yn - 60 # 60 píxeles por encima de la nariz
                
                # Si el usuario sube mucho, evitar que el texto desaparezca por arriba
                if text_y < 30:
                    text_y = 30
                    
                # Fondo negro semitransparente para que el "username" resalte
                cv2.rectangle(bgr_annotated_image, (text_x - 5, text_y - text_size[1] - 5), 
                              (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
                
                # Dibujar el texto en verde
                cv2.putText(bgr_annotated_image, text, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            out.write(bgr_annotated_image)
            cv2.imshow("Pose Detection", bgr_annotated_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Proceso finalizado con éxito. Resultado guardado en: {destination_path}")
        return pull_ups_counter

if __name__ == "__main__":
    counter = PullUpCounter()
    # puedes probar el video en tiempo real:
    counter.process_video(live=True)
    
    # o bien puedes procesar un video guardado:
    # video_path = 'tu_video.mp4'
    # output_dir = 'tu_video_output.mp4'
    # counter.process_video(live=False, video_path=str(video_path), output_path=str(output_dir))