import math
import numpy as np
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python import vision
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
import cv2

def get_joint_angle(p1, p2, p3):
    """
    Calcula el ángulo interno entre tres puntos usando atan2.
    p1 = Hombro, p2 = Codo, p3 = Muñeca
    """
    radians = math.atan2(p3.y - p2.y, p3.x - p2.x) - math.atan2(p1.y - p2.y, p1.x - p2.x)
    angle = abs(math.degrees(radians))
    
    if angle > 180.0:
        angle = 360.0 - angle
        
    return angle

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


class PushUpCounter:

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

        # Definir la ruta de destino si no fue especificada
        if output_path is None:
            if video_path:
                input_path = Path(video_path)
                out_name = f"output_{input_path.stem}.mp4"
            else:
                out_name = "output_pushups_live.mp4"
            destination_path = Path.cwd() / out_name
        else:
            destination_path = Path(output_path).resolve()

        # Notificar al usuario la ruta de destino
        print(f"El video procesado se guardará en: {destination_path}")

        # Configurar el escritor de video (VideoWriter)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or math.isnan(fps):
            fps = 30.0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(destination_path), fourcc, fps, (frame_width, frame_height))

        stage = None
        push_ups_counter = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            detection_result = detector.detect(image)

            left_angle = 0
            right_angle = 0

            if detection_result.pose_landmarks:
                person_detected = detection_result.pose_landmarks[0]
                
                # Puntos del brazo izquierdo
                left_shoulder = person_detected[11]
                left_elbow = person_detected[13]
                left_wrist = person_detected[15]
                
                # Puntos del brazo derecho
                right_shoulder = person_detected[12]
                right_elbow = person_detected[14]
                right_wrist = person_detected[16]
                
                # Calcular los ángulos de ambos brazos
                left_angle = get_joint_angle(left_shoulder, left_elbow, left_wrist)
                right_angle = get_joint_angle(right_shoulder, right_elbow, right_wrist)

            # Lógica de estados estricta (ambos brazos deben cumplir el criterio)
            if left_angle > 150 and right_angle > 150:
                stage = "up"
                
            if left_angle < 130 and right_angle < 130 and stage == "up":
                stage = "down"
                push_ups_counter += 1
                print(f"¡Flexión completada! Total: {push_ups_counter}")

            annotated_image = draw_landmarks_on_image(rgb_frame, detection_result)

            # Convertir imagen anotada de RGB a BGR para OpenCV
            bgr_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            cv2.putText(bgr_annotated_image, f"Push Ups: {push_ups_counter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

            # Escribir frame en el archivo de salida
            out.write(bgr_annotated_image)

            cv2.imshow("Pose Detection", bgr_annotated_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Proceso finalizado con éxito. Resultado guardado en: {destination_path}")
        return push_ups_counter


if __name__ == "__main__":
  desktop = Path.home() / "Desktop"
  # para que el modelo pueda reconocer bien el video es importante que te grabes de frente haciendo las flexiones y tambien no encorvarse demaciado ya que el modelo puede confundirse y no reconocer bien las poses
  # manten la espalda recta y cuello recto
  counter = PushUpCounter()
  # asi procesas el video en tiempo real (necesitas una webcam un celular o alguna camara conectada a tu pc)
  counter.process_video(live=True)
  # si tienes un video pregrabado puedes procesarlo asi:
  # counter.process_video(live=False, video_path=desktop / "push_ups.mp4", output_path=desktop / "output_pushups.mp4")




