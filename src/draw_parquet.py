
import mediapipe as mp
import numpy as np
import pandas as pd
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

def draw(cx, cy, connections, size=512, image=None):
    if image is None:
        image = np.zeros((size, size, 3), dtype=np.uint8)
    
    cx = [int(size*x) for x in cx]
    cy = [int(size*y) for y in cy]
    for l,r in connections:
        cv2.line(image, (cx[l], cy[l]),(cx[r],cy[r]), color=(255, 255, 255))
        cv2.circle(image, (cx[l], cy[l]), radius=2, color=(0,255,0))
        cv2.circle(image, (cx[r], cy[r]), radius=2, color=(0,255,0))
    
    return image

def draw_parquet(path):
    df = pd.read_parquet(path)
    frames = list(df.frame.unique())
    print("frames", frames)
    types = ['left_hand', 'right_hand', 'pose', 'face']
    connections = [mp_holistic.HAND_CONNECTIONS, mp_holistic.HAND_CONNECTIONS, mp_holistic.POSE_CONNECTIONS, mp_holistic.FACEMESH_CONTOURS]
    
    for frame in frames:
        frame_df = df[df.frame == frame]
        image = None
        for type, connection in zip(types, connections):
            part = frame_df[frame_df.type == type]
            part = part.fillna(-1.0)
            image = draw(list(part.x), list(part.y), connection, image=image)
        
        cv2.imshow("image", image)
        cv2.waitKey(-1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)

    args = parser.parse_args()
    draw_parquet(args.path)