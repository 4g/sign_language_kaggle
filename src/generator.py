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

from keypoint_lib import rotate, zoom, shift, hflip, shear, apply_affine_transforms

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
                 cache=True
                 ):

        self.X = None
        self.num_classes = None
        self.labels = None
        self.split_start = split_start
        self.split_end = split_end


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
        self.n_points = 154*2
        
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

    def extract_relevant_parts(self, kps):
        LIP = [
                61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
                291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
                95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
        ]       

        REYE = [
            33, 7, 163, 144, 145, 153, 154, 155, 133,
            246, 161, 160, 159, 158, 157, 173,
        ]

        LEYE = [
            263, 249, 390, 373, 374, 380, 381, 382, 362,
            466, 388, 387, 386, 385, 384, 398,
        ]

        LHAND = list(range(468, 489))
        RHAND = list(range(522, 543))
        POSE = list(range(489, 522))

        # left wrist, right wrist, nose, left eye, right eye, 
        # REFS = [489+15, 489+16, 489, 489+2, 489+5]

        REFS = [500, 501, 512, 513, 159,  386, 13]
        
        kps = kps[:, :, 0:2]

        lip = kps[:, LIP, :]
        lhand = kps[:, LHAND, :]
        rhand = kps[:, RHAND, :]
        pose = kps[:,POSE, :]
        leye = kps[:, LEYE, :]
        reye = kps[:, REYE, :]
        refs = kps[:, REFS, :]
        
        kps = np.concatenate([lip,leye, reye, lhand, rhand, pose, refs], axis=1)    
        return kps


    @staticmethod
    def do_normalise_by_ref(xyz, n_refs=7):
        ref = xyz[:,-n_refs:,:]
        K = ref.shape[-1]
        xyz_flat = ref.reshape(-1,K)
        m = np.nanmean(xyz_flat,0).reshape(1,1,K)
        s = np.nanstd(xyz_flat, 0).mean()
        xyz = xyz - m
        xyz = xyz / s
        return xyz

    def preprocess(self, path):
       
        if path not in self.cache:
            kps = np.load(path)
            kps = kps[:,:,0:2]
            self.cache[path] = kps.astype(np.float16)
        
        kps = self.cache[path]
        
        aug_perc = 0.35
        if self.augment and random.random() < aug_perc:
            idx = random.sample(list(range(len(kps))), min(self.step_size, len(kps)))
            if len(idx) < self.step_size:
                idx = self.reflect_pad(idx, self.step_size)

        else:
            idx = np.linspace(0, len(kps), self.step_size, endpoint=False, dtype=int)
            center = len(kps)//2
            start_idx = center - self.step_size//2
            end_idx = start_idx + self.step_size
            start_idx = max(0, start_idx)
            end_idx = min(end_idx, len(kps))
            idx = np.linspace(start_idx, end_idx, self.step_size, endpoint=False, dtype=int) 

        kps = kps[idx]
        
        # reverse the video
        if self.augment and random.random() < aug_perc:
            kps = kps[::-1]

        
        gr = lambda x : (np.random.random() - .5)*2*x

        aug = 0.3

        if self.augment and random.random() < aug_perc * 2:
            kps = hflip(kps)
            kps = shift(kps, hratio=gr(aug), vratio=gr(aug))
            affines = [rotate(gr(aug)), zoom(gr(aug), gr(aug)), shear(gr(aug), gr(aug))]

            # affines = [rotate(get_random(0.2))]
            if random.random() < aug_perc:
                kps = apply_affine_transforms(kps, affines)

        kps = self.extract_relevant_parts(kps)
        kps = self.do_normalise_by_ref(kps)

        self.n_points = kps.shape[1] * 2
        kps = np.nan_to_num(kps, nan=0.0)
        kps = np.reshape(kps, (-1, self.n_points))
        

        if not self.enable_cache:
            self.cache = {}
        return kps

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
        xs = int(x * 512)
        ys = int(y * 512)
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
        oversample=True
    )

    print(f"Len of Data Generator {len(generator)}")

    for x, y in tqdm(generator):
        pass
        # for frame in x[0]:
        #     # print(y[0], generator.index_to_label[int(y[0])])
        #     img =  draw_points(frame)
        #     cv2.imshow("image", img)
        #     cv2.waitKey(-1)
        # # print(x.shape)
        # print(y.shape)
