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


def show_text(midpoint_coordinates, angle, text_color=(255, 255, 255)):
    cv2.putText(image, str(floor(angle)),
                tuple(np.multiply(midpoint_coordinates, [640, 480]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2, cv2.LINE_AA
                )


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()

        if not success:
            print("Empty Camera Frame")
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
            left_shoulder = extract_landmark_coordinates('LEFT_SHOULDER')
            left_elbow = extract_landmark_coordinates('LEFT_ELBOW')
            left_hip = extract_landmark_coordinates('LEFT_HIP')

            # Calculate Angle
            angle = calculate_angle(left_hip, left_elbow, left_shoulder)

            # Visualize angle
            show_text(left_elbow, 180 - angle)

            # Left Shoulder -> Hip -> Knee
            left_shoulder = extract_landmark_coordinates('LEFT_SHOULDER')
            left_hip = extract_landmark_coordinates('LEFT_HIP')
            left_knee = extract_landmark_coordinates('LEFT_KNEE')

            # Calculate Angle
            angle = calculate_angle(left_shoulder, left_hip, left_knee)

            # Visualize angle
            show_text(left_hip, angle)

            # Left Hip -> Knee -> Ankle
            left_hip = extract_landmark_coordinates('LEFT_HIP')
            left_knee = extract_landmark_coordinates('LEFT_KNEE')
            left_ankle = extract_landmark_coordinates('LEFT_ANKLE')

            # Calculate Angle
            angle = calculate_angle(left_shoulder, left_knee, left_ankle)

            # Visualize angle
            show_text(left_hip, angle)

        except ValueError:
            print("Body Part 404")

        # Subset Connections
        subset_arr = [[11, 23], [23, 25], [25, 27], [11, 13]]

        # Render Detections
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            subset_arr,
            landmark_drawing_spec=None)

        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
