import math

def is_hand_closed(hand_points):
  is_closed = False
  wrist = hand_points[0]

  thumb_cmc = hand_points[1]
  thumb_mcp = hand_points[2]
  thumb_ip = hand_points[3]
  thumb_tip = hand_points[4]

  index_finger_mcp = hand_points[5]
  index_finger_pip = hand_points[6]
  index_finger_dip = hand_points[7]
  index_finger_tip = hand_points[8]

  middle_finger_mcp = hand_points[9]
  middle_finger_pip = hand_points[10]
  middle_finger_dip = hand_points[11]
  middle_finger_tip = hand_points[12]

  ring_finger_mcp = hand_points[13]
  ring_finger_pip = hand_points[14]
  ring_finger_dip = hand_points[15]
  ring_finger_tip = hand_points[16]

  pinky_mcp = hand_points[17]
  pinky_pip = hand_points[18]
  pinky_dip = hand_points[19]
  pinky_tip = hand_points[20]

  """
  Calculates the Euclidean distance between two 2D points using normalized units.
  This acts as a dynamic reference scale to maintain accuracy regardless of 
  how close or far the user's hand is from the physical webcam.
  """
  wrish_middle_mcp_dist = math.dist(
    (wrist.x, wrist.y),
    (middle_finger_mcp.x, middle_finger_mcp.y)
  )
    
  """
  In standard computer graphics coordinate systems, the top-left corner is (0,0).
  Therefore, an increase in the .y parameter indicates spatial movement downwards.
  If the finger tip .y exceeds the knuckle base plus a scalar threshold, 
  it mathematically confirms the finger is curled down into a fist.
  """
  if index_finger_tip.y > index_finger_mcp.y + wrish_middle_mcp_dist * 0.40 and \
    middle_finger_tip.y > middle_finger_mcp.y + wrish_middle_mcp_dist * 0.40 and \
    ring_finger_tip.y > ring_finger_mcp.y + wrish_middle_mcp_dist * 0.40 and \
    pinky_tip.y > pinky_mcp.y + wrish_middle_mcp_dist * 0.40:
    is_closed = True
  return is_closed