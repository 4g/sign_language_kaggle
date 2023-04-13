import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

def sample_points_camera():
  # For webcam input: 
  cap = cv2.VideoCapture(0)

  video_points_array = []

  with mp_holistic.Holistic(
      min_detection_confidence=0.2,
      min_tracking_confidence=0.2) as holistic:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = holistic.process(image)

      # Draw landmark annotation on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      
      p = results.pose_landmarks
      f = results.face_landmarks
      lh = results.left_hand_landmarks
      rh = results.right_hand_landmarks

      n_points = [33, 468, 21, 21]
      all_parts = []

      n = 0
      for idx, part in enumerate([p, f, lh, rh]):
        part_arr = np.zeros((n_points[idx], 3), np.float32)
        part_arr[:] = np.nan

        if part:
          for kpidx, kp in enumerate(part.landmark):
            part_arr[kpidx] = [kp.x, kp.y, kp.z] 

        all_parts.append(part_arr)
      
      all_parts = np.concatenate(all_parts, axis=0)
      video_points_array.append(all_parts)\

      mp_drawing.draw_landmarks(
          image,
          results.face_landmarks,
          mp_holistic.FACEMESH_CONTOURS,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_contours_style())
      mp_drawing.draw_landmarks(
          image,
          results.pose_landmarks,
          mp_holistic.POSE_CONNECTIONS,
          landmark_drawing_spec=mp_drawing_styles
          .get_default_pose_landmarks_style())
      # Flip the image horizontally for a selfie-view display.
      cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
      if cv2.waitKey(5) & 0xFF == 27:
        break
  cap.release()
  return video_points_array

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--out', help='Root dataset directory path', type=str, required=True)
  args = parser.parse_args()

  video_points_array = sample_points_camera()
  np.save(args.out, video_points_array)