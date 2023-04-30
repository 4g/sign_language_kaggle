import numpy as np
import cv2
import albumentations as A
import json

# from https://github.com/tensorflow/tfjs-models/blob/838611c02f51159afdd77469ce67f0e26b7bbb23/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts
keypoint_mapping = json.load(open('src/keypoint_mapping.json'))

def get_keypoint_flip_mapping():
    # return a map of left.right keypoints
    left_hand_points = list(range(468, 489))
    right_hand_points = list(range(522, 543))
    
    pose_left = [1, 2, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    pose_right = [4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]

    pose_left = [489 + x for x in pose_left]
    pose_right = [489 + x for x in pose_right]
    
    swap_ranges = [(left_hand_points, right_hand_points), (pose_left, pose_right)]


    right_eye = ['rightEyeUpper0', 'rightEyeLower0', 'rightEyeUpper1', 'rightEyeLower1', 'rightEyeUpper2', 'rightEyeLower2', 'rightEyeLower3', 'rightEyebrowUpper', 'rightEyebrowLower', 'rightEyeIris','noseRightCorner', 'rightCheek']
    left_eye = [x.replace('right', 'left') for x in right_eye]


    for p1, p2 in zip(left_eye, right_eye):
        r = keypoint_mapping[p1], keypoint_mapping[p2]
        swap_ranges.append(r)


    # these parts dont have different left right parts, but have sides 
    one_part = ['lipsUpperOuter', 'lipsLowerOuter','lipsUpperInner', 'lipsLowerInner'] 
    for part in one_part:
        indices = keypoint_mapping[part]
        n_indices = len(indices)
        right_indices = indices[:n_indices//2]
        left_indices = indices[::-1][:n_indices//2]
        swap_ranges.append((left_indices, right_indices))
    
    left = []
    right = []
    for l,r in swap_ranges:
        left.extend(l)
        right.extend(r)
    
    return left, right 

mapping = get_keypoint_flip_mapping()
left_points = mapping[0]
right_points = mapping[1]

def get_part_ranges(): 
    x = {
        "face" : (0, 468),
        "left_hand" : (468, 489),
        "pose" : (489, 522),
        "right_hand": (522, 543)
    }
    
    return x

def normalize_by_shoulders(keypoints, lshoulder_index, rshoulder_index):
    keypoints = keypoints.copy()
    
    desired_shoulder_length = 2.0

    for frame_index in range(len(keypoints)):
        lshoulder = keypoints[frame_index, lshoulder_index, :]
        rshoulder = keypoints[frame_index, rshoulder_index, :]
        shoulder_length = np.linalg.norm(lshoulder - rshoulder)
        
        resize_ratio = desired_shoulder_length/shoulder_length
        
        keypoints[frame_index] *= resize_ratio

        lshoulder = keypoints[frame_index, lshoulder_index, :]
        rshoulder = keypoints[frame_index, rshoulder_index, :]
        shoulder_center = (lshoulder + rshoulder) / 2
        keypoints[frame_index] = keypoints[frame_index] - shoulder_center

    # keypoints += 0.5
    return keypoints


def normalize_z(zkeypoints, lshoulder_index, rshoulder_index):
    keypoints = zkeypoints.copy()
    keypoints = (keypoints - np.nanmin(keypoints))/(np.nanmax(keypoints) - np.nanmin(keypoints))
    for frame_index in range(len(keypoints)):
        lshoulder = keypoints[frame_index, lshoulder_index]
        rshoulder = keypoints[frame_index, rshoulder_index]
        shoulder_center = (lshoulder + rshoulder) / 2
        keypoints[frame_index] = keypoints[frame_index] - shoulder_center

    return keypoints



def zflip(zkeypoints):

    # indices are ready 
    # lets make flipped keypoints now

    keypoints = zkeypoints.copy()
    # these parts have left right swapped
    
    tmp = keypoints[:,left_points]
    keypoints[:,left_points] = keypoints[:,right_points]
    keypoints[:,right_points] = tmp 

    return keypoints

def hflip(keypoints):
   

    # indices are ready 
    # lets make flipped keypoints now

    keypoints = keypoints.copy()
    keypoints[:,:,0] = 1 - keypoints[:,:,0]
    
    # these parts have left right swapped
    
    tmp = keypoints[:,left_points]
    keypoints[:,left_points] = keypoints[:,right_points]
    keypoints[:,right_points] = tmp 


    return keypoints

def apply_affine_transforms(keypoints, transforms):
    n_frames = len(keypoints)
    n_points = keypoints.shape[1]

    keypoints = np.reshape(keypoints, (-1, 2))
    keypoints = keypoints * 512
    keypoints = np.concatenate([keypoints, np.ones((len(keypoints), 1), dtype=np.float32)], axis=-1)


    for M in transforms:
        keypoints = M.dot(keypoints.T)
        keypoints = keypoints.T

    kps = keypoints[:,0:2]/512
    kps = np.reshape(kps, (n_frames, n_points, 2))
    return kps

def zoom(hratio, vratio):
    M = np.eye(2)
    M[1][1] = vratio + 1
    M[0][0] = hratio + 1
    return M

def shear(hratio, vratio):
    M = np.eye(2)
    M[0][1] = hratio
    M[1][0] = vratio
    return M

def rotate(ratio):
    angle = ratio * 45
    M = cv2.getRotationMatrix2D((512 / 2, 512 / 2), angle, 1)
    return M


def shift(keypoints, vratio, hratio):
    keypoints = keypoints.copy()
    keypoints[:,:,0] += hratio
    keypoints[:,:,1] += vratio
    return keypoints


def point_to_part(index):
    if index in left_points:
        return 0
    if index in right_points:
        return 1
    else:
        return 2

def draw_points(points, normalized=False):
    points = np.nan_to_num(points, 0.0)
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    colors = [(0, 0, 255), (0, 255, 0), (255, 255, 255)]

    index = 0

    for x,y in zip(points[:, 0], points[:, 1]):
        xs = int(x * 512) + (256 if normalized else 0)
        ys = int(y * 512) + (256 if normalized else 0)
        
        cv2.circle(img, (xs, ys), radius=1, color=colors[point_to_part(index)], thickness=-1)
        index += 1

    return img


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='Root dataset directory path', type=str, required=True)
    args = parser.parse_args()

    data_dir = args.data_dir
    data_dir = Path(data_dir)
    npy_dir = data_dir / "train_npy"
    files = list(npy_dir.glob("*.npy"))
    print("nfiles", len(files))
    
    for file in files:
        allkps = np.load(file)
        allkps = allkps[:, :, 0:2]

        gr = lambda x : (np.random.random() - .5)*2*x

        rot_kps = hflip(allkps)
        rot_kps = shift(rot_kps, hratio=gr(0.2), vratio=gr(0.2))
        affines = [rotate(gr(.1)), zoom(gr(.1), gr(.1)), shear(gr(.1), gr(.1))]
        # affines = [zoom(0.2, 0.2)]
        # rot_kps = allkps
        rot_kps = apply_affine_transforms(rot_kps, affines)

        rot_kps = normalize_by_shoulders(keypoints=rot_kps, lshoulder_index=500, rshoulder_index=501)
        

        for index, kps in enumerate(allkps):
            img = draw_points(kps)
            rot_img = draw_points(rot_kps[index], normalized=True)

            cv2.imshow("img", img)
            cv2.imshow("rot_img", rot_img)
            cv2.waitKey(-1)