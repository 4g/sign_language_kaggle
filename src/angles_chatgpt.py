
import cv2
import mediapipe as mp
import math
import numpy as np

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Initialize Holistic model
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the image horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe Holistic model
    results = holistic.process(frame_rgb)

    # # Draw the landmarks on the image
    # mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
    # mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    # mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    # mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    # Get the landmarks of the left and right hand
    left_hand_landmarks = results.left_hand_landmarks
    right_hand_landmarks = results.right_hand_landmarks

    if left_hand_landmarks is not None:
        # Extract the x,y coordinates of the landmarks for the left hand
        coords = np.array([(lmk.x, lmk.y) for lmk in left_hand_landmarks.landmark])

        # Calculate the angles between adjacent finger joints for the left hand
        angles = np.degrees(np.arctan2(np.diff(coords[:, 1]), np.diff(coords[:, 0])))

        # Append 0 for the thumb angle (thumb has only 3 joints instead of 4 or 5)
        angles = np.append(angles, 0)

        print("Left hand angles:", angles)

    if right_hand_landmarks is not None:
        # Extract the x,y coordinates of the landmarks for the right hand
        coords = np.array([(lmk.x, lmk.y) for lmk in right_hand_landmarks.landmark])

        # Calculate the angles between adjacent finger joints for the right hand
        angles = np.degrees(np.arctan2(np.diff(coords[:, 1]), np.diff(coords[:, 0])))

        # Append 0 for the thumb angle (thumb has only 3 joints instead of 4 or 5)
        angles = np.append(angles, 0)

        print("Right hand angles:", angles)
