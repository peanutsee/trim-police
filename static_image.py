import cv2
import mediapipe as mp
import numpy as np
from math import floor

# Utils Function
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Calculate Angle in Radians
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])

    # Calculate Angle in Degrees
    degrees = np.abs(radians * 180.0 / np.pi)

    if degrees > 180:
        degrees = 360 - degrees

    return degrees


def extract_landmark_coordinates(body_part):
    position = [landmarks[mp_pose.PoseLandmark[body_part].value].x,
                landmarks[mp_pose.PoseLandmark[body_part].value].y]
    return position


def show_text(midpoint_coordinates, angle, translate_x, translate_y, text_color=(0, 0, 255)):
    cv2.putText(image, str(floor(angle)),
                tuple(np.multiply(midpoint_coordinates, [translate_x, translate_y]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA
                )


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For static images:
IMAGE_FILES = ['trim-test-1.jpg']
BG_COLOR = (192, 192, 192) # gray
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)

    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
      continue

    image.flags.writeable = False

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Make Detections
    results = pose.process(image)

    # Convert RGB to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Extract Landmarks
    try:
        landmarks = results.pose_landmarks.landmark

        # Left Hip -> Shoulder -> Elbow
        RIGHT_SHOULDER = extract_landmark_coordinates('RIGHT_SHOULDER')
        RIGHT_ELBOW = extract_landmark_coordinates('RIGHT_ELBOW')
        RIGHT_HIP = extract_landmark_coordinates('RIGHT_HIP')

        # Calculate Angle
        angle = calculate_angle(RIGHT_HIP, RIGHT_ELBOW, RIGHT_SHOULDER)

        # Visualize angle
        show_text(RIGHT_ELBOW, 180 - angle, 270, 120)

        # Left Shoulder -> Hip -> Knee
        RIGHT_SHOULDER = extract_landmark_coordinates('RIGHT_SHOULDER')
        RIGHT_HIP = extract_landmark_coordinates('RIGHT_HIP')
        RIGHT_KNEE = extract_landmark_coordinates('RIGHT_KNEE')

        # Calculate Angle
        angle = calculate_angle(RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE)

        # Visualize angle
        show_text(RIGHT_HIP, angle, 270, 150)

        # Left Hip -> Knee -> Ankle
        RIGHT_HIP = extract_landmark_coordinates('RIGHT_HIP')
        RIGHT_KNEE = extract_landmark_coordinates('RIGHT_KNEE')
        RIGHT_ANKLE = extract_landmark_coordinates('RIGHT_ANKLE')

        # Calculate Angle
        angle = calculate_angle(RIGHT_SHOULDER, RIGHT_KNEE, RIGHT_ANKLE)

        # Visualize angle
        show_text(RIGHT_HIP, angle, 100, 170)

    except ValueError:
        print("Body Part 404")

    # Subset Connections
    subset_arr = [[12, 24], [24, 26], [26, 28], [12, 14]]

    # Render Detections
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        subset_arr,
        landmark_drawing_spec=None)

    cv2.imshow('Trim Right Diver Profile', image)
    cv2.waitKey(0)


