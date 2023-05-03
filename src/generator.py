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

from keypoint_lib import rotate, zoom, shift, shear, apply_affine_transforms, rotate_3d 

class BinaryGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 data_dir,
                 batch_size = 64,
                 step_size = 64,
                 shuffle = False,
                 split_start=0.0,
                 split_end=1.0,
                 oversample=True,
                 augment=True,
                 cache=True,
                 ):
        
        random.seed(42)   
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
        self.calls = 0
        
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
        self.LHAND = list(range(468, 489))
        self.RHAND = list(range(522, 543))
        # self.n_points = len(self.relevant_indices)*2
        self.n_points = 152
        
        data_dir = Path(data_dir)
        npy_dir = data_dir / "train_npy"
        training_csv = data_dir / "train.csv"
        train_data = pd.read_csv(training_csv)
        label_map = json.load(open(data_dir / "sign_to_prediction_index_map.json"))
        self.num_classes = len(label_map)
        
        participants = train_data.participant_id.unique()
        # random.shuffle(participants)

        start, end = int(self.split_start * len(participants)) , int(self.split_end * len(participants))
        chosen_participants = set(participants[start:end])
        
        for elem in train_data.to_dict(orient='records'):
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
        self.calls += 1
        indices = self.indices[n * self.batch_size: n * self.batch_size + self.batch_size]
        x_batch = np.zeros((len(indices), self.step_size, 543, 3), dtype=np.float32)
        y_batch = np.zeros((len(indices), self.num_classes), dtype=np.float32)
        

        for index, rindex in enumerate(indices):
            sample = self.preprocess(self.X[rindex])
            x_batch[index] = sample
            y_batch[index][self.labels[rindex]] = 1.0
            
        return x_batch, y_batch


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

    def preprocess(self, path):
        if path not in self.cache:
            kps = np.load(path)
            self.cache[path] = kps.astype(np.float32)

        kps = self.cache[path]
        
        nonna_kps = []
        # print("-----")
        for i, frame in enumerate(kps):
            # print(frame[self.LHAND][0][0], frame[self.RHAND][0][0])
            lna = np.isnan(frame[self.LHAND][0][0])
            rna = np.isnan(frame[self.RHAND][0][0])
            if lna and rna:
                continue
            nonna_kps.append(i)

        kps = kps[nonna_kps]
        
        aug_perc = 0.35
        aug_strength = 0.3    

        if self.augment and (random.random() < aug_perc*2):
            if random.random() < 0.5:
                idx = list(range(len(kps)))
                if len(kps) < self.step_size:
                    idx = self.reflect_pad(idx, self.step_size)
                else:
                    idx = random.sample(list(range(len(kps))), self.step_size)
                    idx = sorted(idx)
                    # idx = np.linspace(min(idx), max(idx), self.step_size, endpoint=True, dtype=int)

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
        
        # reverse the video
        if self.augment and (random.random() < aug_perc):
            kps = kps[::-1]
            
        gr = lambda x : (np.random.random() - .5)*2*x

        # separate z because it is augmented differently
        zkps = kps[:,:,2:]
        kps = kps[:,:,0:2]


        if self.augment and (random.random() < aug_perc * 2):
            # kps = hflip(kps)

            # zkps = zflip(zkps)
            kps = shift(kps, hratio=gr(aug_strength), vratio=gr(aug_strength))

            # affines = [rotate(get_random(0.2))]
            if random.random() < aug_perc*2:
                affines = [rotate(gr(aug_strength)), zoom(gr(aug_strength), gr(aug_strength)), shear(gr(aug_strength), gr(aug_strength))]
                kps = apply_affine_transforms(kps, affines)

            
        kpsz = np.concatenate([kps, zkps], axis=-1)
        return kpsz

    def on_epoch_end(self):
        """
        Runs at end of every epoch
        """
        # print("Cache size", len(self.cache), "calls", self.calls)
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
        oversample=False,
        split_start=0.9,
        split_end=1.0
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
