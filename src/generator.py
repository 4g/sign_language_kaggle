import random
import cv2
import json
import albumentations as A
import numpy as np
import tensorflow as tf
from pathlib import Path
from collections import Counter
from typing import List, Tuple

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import interpolate

from keypoint_lib import rotate, zoom, shift, hflip, shear, apply_affine_transforms, normalize_by_shoulders

class BinaryGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 data_dir,
                 batch_size: int = 64,
                 step_size: int = 64,
                 shuffle: bool = False,
                 split_start=0.0,
                 split_end=1.0,
                 oversample=True,
                 augment=True,
                 cache=True,
                 mode='train'
                 ):

        self.X = None
        self.num_classes = None
        self.labels = None
        self.split_start = split_start
        self.split_end = split_end
        self.mode = mode


        self.batch_size = batch_size
        self.step_size = step_size
        self.shuffle = shuffle
        self.oversample = oversample
        self.augment = augment
        self.cache = {}        
        self.indices = []
        self.enable_cache = cache
        
        self.load_data(data_dir)
        
        self.on_epoch_end()

    def __len__(self) -> int:
        return len(self.indices) // self.batch_size
    
    
    def rebalance(self):
        """
        Rebalance the dataset by labels
        Oversample the indices which have rare labels.
        After rebalance all labels have almost equal counts.
        """

        labels_count = Counter(self.labels)
        # print("Initial count of labels", labels_count)
        labels = np.asarray(self.labels)
        max_count = max(labels_count.values())
        resampled_indices = []
        for label in labels_count.keys():
            label_indices = list(np.where(labels == label)[0])
            size_diff = max_count - labels_count[label]
            new_indices = []
            while size_diff > 0:
                # print(size_diff)
                new_indices = random.sample(label_indices, min(size_diff, len(label_indices)))
                size_diff -= len(new_indices)
                resampled_indices.extend(new_indices)

        self.indices = list(range(len(labels)))
        if self.oversample:
            self.indices += resampled_indices
        final_count = Counter(np.asarray(self.labels)[self.indices])
        # print("final count of labels", final_count)

    def load_data(self, data_dir):
        self.X = []
        self.labels = []
        self.make_relevant_parts()
        # self.n_points = len(self.relevant_indices)*2
        self.n_points = 314
        
        data_dir = Path(data_dir)
        npy_dir = data_dir / "train_npy"
        training_csv = data_dir / "train.csv"
        train_data = pd.read_csv(training_csv)
        label_map = json.load(open(data_dir / "sign_to_prediction_index_map.json"))
        self.num_classes = len(label_map)
        
        participants = train_data.participant_id.unique()
        # random.shuffle(participants)

        start, end = int(self.split_start * len(participants)) , int(self.split_end * len(participants))
        print(start, end)
        chosen_participants = set(participants[start:end])
        print(chosen_participants)
        
        for elem in tqdm(train_data.to_dict(orient='records'), "loading data"):
            label = elem["sign"]
            label = label_map[label]
            sequence_id = elem["sequence_id"]
            participant_id = elem["participant_id"]
            
            if participant_id in chosen_participants:
                npy_path = npy_dir / f"{participant_id}_{sequence_id}.npy"

                self.X.append(npy_path)
                self.labels.append(label)
        
        self.index_to_label = {}
        for label in label_map:
            self.index_to_label[label_map[label]] = label

    def __getitem__(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        indices = self.indices[n * self.batch_size: n * self.batch_size + self.batch_size]
        x_batch = np.zeros((self.batch_size, self.step_size, self.n_points), dtype=np.float16)
        y_batch = np.zeros((self.batch_size, self.num_classes), dtype=np.float16)
        # y_batch = np.zeros(self.batch_size, dtype=np.float16)
        

        for index, rindex in enumerate(indices):
            sample = self.preprocess(self.X[rindex])
            x_batch[index] = sample
            y_batch[index][self.labels[rindex]] = 1.0
            # y_batch[index] = self.labels[rindex]

        return (x_batch, y_batch)
    

    def reflect_pad(self, arr:list, desired_length:int):
        """
        This is a way to pad the array without breaking continuity.
        Reflect the array on itself till it reaches desired length

        e.g.  arr = [0,1,2] , desired_length = 10
        arr = arr + arr[::-1] = [0,1,2,2,1,0]
        arr = arr + arr[::-1] = [0,1,2,2,1,0,0,1,2,2,1,0]
        len(arr) = 12
        arr[:desired_length] = [0,1,2,2,1,0,0,1,2,2]
        """
        while len(arr) < desired_length:
            arr = arr + arr[::-1]
        
        return arr[:desired_length]

    def resize_array(self, arr, size):
        if len(arr) == 1:
            arr = arr * size
            return arr
            
        old_size = len(arr)
        x = np.linspace(0, 1, num=old_size)
        f = interpolate.interp1d(x=x, y=arr)
        new_points = np.linspace(0, 1, num=size)
        new_arr = f(new_points)

        return new_arr

    def make_relevant_parts(self):
        self.LHAND = list(range(468, 489))
        self.RHAND = list(range(522, 543))
        self.POSE = list(range(489, 522))
        self.SHOULDERS = [500, 501]
        self.ELBOWS = [502, 503]
        self.WRISTS = [489+15, 489+16]
        self.HAND_POSE_POINTS = [489+x for x in [15,16,17,18,19,20,21,22]]
        self.FINGERS = [468 + x for x in [0,1,4,5,8,9,12,13,16,17,20]] + [522 + x for x in [0,1,4,5,8,9,12,13,16,17,20]]

        self.UPPER_LIP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415]
        
        self.LOWER_LIP = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324]
        
        self.LIP = self.LOWER_LIP + self.UPPER_LIP        
        

        # self.LIP = [82, 13, 
        #             87, 14]
        
        self.REYE = [
            145, 153,
            158, 157,
        ]

        self.LEYE = [
            374, 380,
            385, 384
        ]
        
        # self.relevant_parts = [self.LEYE, self.REYE, self.LIP, self.LHAND, self.RHAND, self.SHOULDERS, self.ELBOWS, self.HAND_POSE_POINTS]
        self.relevant_parts = [self.LEYE, self.REYE, self.LIP, self.FINGERS, self.SHOULDERS, self.ELBOWS, self.HAND_POSE_POINTS, self.LHAND, self.RHAND]

        self.relevant_indices = []
        for part in self.relevant_parts:
            self.relevant_indices += part


    def extract_relevant_parts(self, kps):
        kps = kps[:, self.relevant_indices]
        return kps


    def preprocess(self, path):
       
        if path not in self.cache:
            kps = np.load(path)
            self.cache[path] = kps.astype(np.float16)

        kps = self.cache[path]
        zkps = kps[:,:,2]
        kps = kps[:,:,0:2]
        aug_perc = 0.35

        # nonna_kps = []
        # # print("-----")
        # for i, frame in enumerate(kps):
        #     # print(frame[self.LHAND][0][0], frame[self.RHAND][0][0])
        #     lna = np.isnan(frame[self.LHAND][0][0])
        #     rna = np.isnan(frame[self.RHAND][0][0])
        #     if lna and rna:
        #         continue
        #     nonna_kps.append(i)
        
        # kps = kps[nonna_kps]

        if self.augment and random.random() < aug_perc*2:
            if random.random() < 0.5:
                idx = random.sample(list(range(len(kps))), min(self.step_size, len(kps)))
                # if len(idx) < self.step_size:
                #     idx = self.reflect_pad(idx, self.step_size)
                idx = np.linspace(min(idx), max(idx), self.step_size, endpoint=True, dtype=int)

            else:
                center = len(kps)//2
                start_idx = center - self.step_size//2
                end_idx = start_idx + self.step_size
                start_idx = max(0, start_idx)
                end_idx = min(end_idx, len(kps))
                idx = np.linspace(start_idx, end_idx, self.step_size, endpoint=False, dtype=int)
        else:
            idx = np.linspace(0, len(kps), self.step_size, endpoint=False, dtype=int)

        kps = kps[idx]
        zkps = zkps[idx]
        
        # reverse the video
        if self.augment and random.random() < aug_perc:
            kps = kps[::-1]

        
        gr = lambda x : (np.random.random() - .5)*2*x

        aug = 0.3

        if self.augment and random.random() < aug_perc * 2:
            kps = hflip(kps)
            kps = shift(kps, hratio=gr(aug), vratio=gr(aug))

            # affines = [rotate(get_random(0.2))]
            if random.random() < aug_perc:
                affines = [rotate(gr(aug)), zoom(gr(aug), gr(aug)), shear(gr(aug), gr(aug))]
                kps = apply_affine_transforms(kps, affines)

        angles = []
        langles = self.angles_between(kps[:,self.LHAND,:])
        rangles = self.angles_between(kps[:,self.RHAND,:])
        pangles = self.angles_between(kps[:,self.HAND_POSE_POINTS,:])
        # fangles = self.angles_between(kps[:,self.FINGERS,:])
        handz = zkps[:, self.LHAND + self.RHAND]
        hand_pos = self.hand_neck_distance(kps)
        lip_openings = self.lip_opening(kps)


        angles = np.concatenate([langles, rangles, pangles, handz, hand_pos, lip_openings], axis=-1)


        kps = normalize_by_shoulders(kps, lshoulder_index=500, rshoulder_index=501)
        
        kps = self.extract_relevant_parts(kps)
        # mv = self.extract_relevant_parts(mv)

        kps = np.reshape(kps, (-1, kps.shape[1] * 2))
        
        if not self.enable_cache:
            self.cache = {}
        
        kps = np.concatenate([kps, hand_pos, angles], axis=-1)
        kps = np.nan_to_num(kps, nan=0.0)
        return kps

    def lip_opening(self, kps):
        r  = []
        for frame_index in range(len(kps)):
            d = []
            for p1, p2 in zip(self.UPPER_LIP, self.LOWER_LIP):
                l = np.linalg.norm(kps[frame_index, p1, :] - kps[frame_index, p2, :])
                d.append(l)
            r.append(d)
        return np.asarray(r, np.float16)


    def hand_neck_distance(self, kps):
        lshoulder_index = self.SHOULDERS[0]
        rshoulder_index = self.SHOULDERS[1]
        dists = []
        for frame_index in range(len(kps)):
            lshoulder = kps[frame_index, lshoulder_index, :]
            rshoulder = kps[frame_index, rshoulder_index, :]
            shoulder_center = (lshoulder + rshoulder) / 2
            lwrist = kps[frame_index, self.WRISTS[0]]
            rwrist = kps[frame_index, self.WRISTS[1]]
            d1 = np.linalg.norm(lwrist - shoulder_center)
            d2 = np.linalg.norm(rwrist - shoulder_center)
            dists.append([d1, d2])
        
        dists = np.asarray(dists, dtype=np.float16)
        return dists

    def angles_between(self, kps):
        angles = np.arctan2(np.diff(kps[:, :, 1], axis=1), np.diff(kps[:, :, 0], axis=1))
        angles = np.concatenate([angles, np.zeros((angles.shape[0], 1))], axis=1)
        return angles

    def motion_vector(self, kps):
        mv = np.diff(kps, axis=0)
        mv_norm = np.linalg.norm(mv,  axis=2)
        mv_norm = np.concatenate([mv_norm, mv_norm[-1:]], axis=0)
        
        # mv_norm = np.expand_dims(mv_norm, axis=-1)
        return mv_norm


    def on_epoch_end(self):
        """
        Runs at end of every epoch
        """
        
        self.rebalance()

        if self.shuffle:
            random.shuffle(self.indices)


def draw_points(points):
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    n_points = len(points) // 2
    points = np.reshape(points, (n_points, 2))
    for x,y in zip(points[:, 0], points[:, 1]):
        xs = int(x * 512) + 256
        ys = int(y * 512) + 256
        cv2.circle(img, (xs, ys), radius=2, color=(255,255,255), thickness=-1)
    
    return img


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='Root dataset directory path', type=str, required=True)
    args = parser.parse_args()
    

    generator = BinaryGenerator(
        args.data_dir,
        batch_size=256,
        step_size=64,
        shuffle=True,
        augment=True,
        oversample=False
    )

    print(f"Len of Data Generator {len(generator)}")

    for x, y in tqdm(generator):
        # pass
        for frame in x[0]:
            # print(y[0], generator.index_to_label[int(y[0])])
            img =  draw_points(frame)
            cv2.imshow("image", img)
            cv2.waitKey(-1)
        print(x.shape)
        print(y.shape)
