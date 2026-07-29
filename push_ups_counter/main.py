
import math
import numpy as np
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python import vision
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
import cv2
from push_ups_counter import count_push_up

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


# Initialize PoseLandmarker outside the loop
model_path = Path(__file__).parent.parent / 'models' / 'pose_landmarker.task'
base_options = python.BaseOptions(model_asset_path=str(model_path))

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)

detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
stage = None
push_ups_counter = 0
while True:
  ret, frame = cap.read()
  if not ret:
    break

  rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

  detection_result = detector.detect(image)
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
        
        # Opcional: imprimir ambos para calibrar
        # print(f"Izq: {int(left_angle)} | Der: {int(right_angle)}")
        
        # Lógica de estados estricta (ambos brazos deben cumplir el criterio)
        if left_angle > 160 and right_angle > 160:
            stage = "up"
            
        if left_angle < 130 and right_angle < 130 and stage == "up":
            stage = "down"
            push_ups_counter += 1
            print(f"¡Flexión completada! Total: {push_ups_counter}")

  annotated_image = draw_landmarks_on_image(rgb_frame, detection_result)

  # Convert RGB annotated image back to BGR for cv2.imshow
  bgr_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
  cv2.putText(bgr_annotated_image, "Push Ups: " + str(push_ups_counter), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
  cv2.imshow("Pose Detection", bgr_annotated_image)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()

