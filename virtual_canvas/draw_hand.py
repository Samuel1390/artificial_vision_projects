import cv2 as cv
import math

def draw_hand(frame, hand_points, skeleton_width, point_radius):
  """
  Extracts the dimensional shape (height, width, channels) of the frame array.
  OpenCV represents images as standard NumPy ndarrays where shape properties
  are accessed via a tuple of dimensions.
  """
  h, w, _ = frame.shape
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

  for i in range(20):
    """
    MediaPipe outputs coordinates normalized between 0.0 and 1.0 relative to image bounds.
    To draw them on the screen, they are multiplied by the pixel width and height
    and cast to absolute integer pixel locations.
    """
    point_x, point_y = int(hand_points[i].x * w), int(hand_points[i].y * h)
    """
    cv.circle modifies the frame array in-place. It takes the target image,
    center coordinates as a tuple, radius, color in BGR format (0, 0, 255 for Red),
    and a thickness of -1 which instructs OpenCV to fill the circle entirely.
    """
    cv.circle(frame, (point_x, point_y), point_radius, (0, 0, 255), -1)

  """
  cv.line draws a straight vector segment on the image matrix. It connects 
  two pixel coordinate tuples using a specified BGR color value (255, 255, 255 for White)
  and a line thickness in pixels.
  """
  cv.line(frame, (int(wrist.x * w), int(wrist.y * h)), (int(thumb_cmc.x * w), int(thumb_cmc.y * h)), (255, 255, 255), skeleton_width)
  cv.line(frame, (int(thumb_cmc.x * w), int(thumb_cmc.y * h)), (int(thumb_mcp.x * w), int(thumb_mcp.y * h)), (255, 255, 255), skeleton_width)
  cv.line(frame, (int(thumb_mcp.x * w), int(thumb_mcp.y * h)), (int(thumb_ip.x * w), int(thumb_ip.y * h)), (255, 255, 255), skeleton_width)
  cv.line(frame, (int(thumb_ip.x * w), int(thumb_ip.y * h)), (int(thumb_tip.x * w), int(thumb_tip.y * h)), (255, 255, 255), skeleton_width)

  cv.line(frame, (int(wrist.x * w), int(wrist.y * h)), (int(index_finger_mcp.x * w), int(index_finger_mcp.y * h)), (255, 255, 255), skeleton_width)
  cv.line(frame, (int(index_finger_mcp.x * w), int(index_finger_mcp.y * h)), (int(index_finger_pip.x * w), int(index_finger_pip.y * h)), (255, 255, 255), skeleton_width)
  cv.line(frame, (int(index_finger_pip.x * w), int(index_finger_pip.y * h)), (int(index_finger_dip.x * w), int(index_finger_dip.y * h)), (255, 255, 255), skeleton_width)
  cv.line(frame, (int(index_finger_dip.x * w), int(index_finger_dip.y * h)), (int(index_finger_tip.x * w), int(index_finger_tip.y * h)), (255, 255, 255), skeleton_width)

  cv.line(frame, (int(wrist.x * w), int(wrist.y * h)), (int(middle_finger_mcp.x * w), int(middle_finger_mcp.y * h)), (255, 255, 255), skeleton_width)
  cv.line(frame, (int(middle_finger_mcp.x * w), int(middle_finger_mcp.y * h)), (int(middle_finger_pip.x * w), int(middle_finger_pip.y * h)), (255, 255, 255), skeleton_width)
  cv.line(frame, (int(middle_finger_pip.x * w), int(middle_finger_pip.y * h)), (int(middle_finger_dip.x * w), int(middle_finger_dip.y * h)), (255, 255, 255), skeleton_width)
  cv.line(frame, (int(middle_finger_dip.x * w), int(middle_finger_dip.y * h)), (int(middle_finger_tip.x * w), int(middle_finger_tip.y * h)), (255, 255, 255), skeleton_width)

  cv.line(frame, (int(wrist.x * w), int(wrist.y * h)), (int(ring_finger_mcp.x * w), int(ring_finger_mcp.y * h)), (255, 255, 255), skeleton_width)
  cv.line(frame, (int(ring_finger_mcp.x * w), int(ring_finger_mcp.y * h)), (int(ring_finger_pip.x * w), int(ring_finger_pip.y * h)), (255, 255, 255), skeleton_width)
  cv.line(frame, (int(ring_finger_pip.x * w), int(ring_finger_pip.y * h)), (int(ring_finger_dip.x * w), int(ring_finger_dip.y * h)), (255, 255, 255), skeleton_width)
  cv.line(frame, (int(ring_finger_dip.x * w), int(ring_finger_dip.y * h)), (int(ring_finger_tip.x * w), int(ring_finger_tip.y * h)), (255, 255, 255), skeleton_width)

  cv.line(frame, (int(wrist.x * w), int(wrist.y * h)), (int(pinky_mcp.x * w), int(pinky_mcp.y * h)), (255, 255, 255), skeleton_width)
  cv.line(frame, (int(pinky_mcp.x * w), int(pinky_mcp.y * h)), (int(pinky_pip.x * w), int(pinky_pip.y * h)), (255, 255, 255), skeleton_width)
  cv.line(frame, (int(pinky_pip.x * w), int(pinky_pip.y * h)), (int(pinky_dip.x * w), int(pinky_dip.y * h)), (255, 255, 255), skeleton_width)
  cv.line(frame, (int(pinky_dip.x * w), int(pinky_dip.y * h)), (int(pinky_tip.x * w), int(pinky_tip.y * h)), (255, 255, 255), skeleton_width)