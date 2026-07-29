import math

def count_push_up(person_detected, last_person_detected):
  EXTENSION_THRESHOLD = 10
  FLEX_THRESHOLD = 40
  
  left_arm_angle, right_arm_angle = detection_result_info(person_detected)
  last_left_arm_angle, last_right_arm_angle = detection_result_info(last_person_detected)

  if not (all((left_arm_angle, right_arm_angle, last_left_arm_angle, last_right_arm_angle))) or not last_person_detected or not person_detected:
    return {"left_arm": 0, "right_arm": 0}

  if math.degrees(last_left_arm_angle) < EXTENSION_THRESHOLD and math.degrees(left_arm_angle) > FLEX_THRESHOLD and math.degrees(last_right_arm_angle) < EXTENSION_THRESHOLD and math.degrees(right_arm_angle) > FLEX_THRESHOLD:
    return {"left_arm": 1, "right_arm": 1}
  elif math.degrees(last_left_arm_angle) < EXTENSION_THRESHOLD and math.degrees(left_arm_angle) > FLEX_THRESHOLD:
    return {"left_arm": 1, "right_arm": 0}
    print("Left")
  elif math.degrees(last_right_arm_angle) < EXTENSION_THRESHOLD and math.degrees(right_arm_angle) > FLEX_THRESHOLD:
    print("Right")
    return {"left_arm": 0, "right_arm": 1}
  else:
    return {"left_arm": 0, "right_arm": 0}
  

def get_line_slope(p1, p2):
  return (p2.y - p1.y) / (p2.x - p1.x)

def get_lines_angle(slope1, slope2):
  return abs(math.atan((slope1 - slope2) / (1 + slope1 * slope2)))


def detection_result_info(person_detected):

  if not person_detected:
    return None, None

  left_shoulder_idx = 11
  right_shoulder_idx = 12
  left_elbow_idx = 13
  right_elbow_idx = 14
  left_wrist_idx = 15
  right_wrist_idx = 16

  left_shoulder = person_detected[left_shoulder_idx]
  right_shoulder = person_detected[right_shoulder_idx]
  left_elbow = person_detected[left_elbow_idx]
  right_elbow = person_detected[right_elbow_idx]
  left_wrist = person_detected[left_wrist_idx]
  right_wrist = person_detected[right_wrist_idx]

  left_arm_slope = get_line_slope(left_shoulder, left_elbow)
  right_arm_slope = get_line_slope(right_shoulder, right_elbow)
  left_forearm_slope = get_line_slope(left_elbow, left_wrist)
  right_forearm_slope = get_line_slope(right_elbow, right_wrist)

  left_arm_angle = get_lines_angle(left_arm_slope, left_forearm_slope)
  right_arm_angle = get_lines_angle(right_arm_slope, right_forearm_slope)

  return left_arm_angle, right_arm_angle
  