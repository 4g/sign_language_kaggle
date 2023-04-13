from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

def pandas_fill(arr):
    df = pd.DataFrame(arr)
    df.fillna(method='ffill', axis=0, inplace=True)
    out = df.values
    return out


def get_keypoints(df):
    arr = df[['x','y','z']].values
    n_frames = arr.shape[0] // 543
    arr = np.reshape(arr, (n_frames, 543 * 3))
    arr = pandas_fill(arr)
    return arr

def iter_data(data_dir):
    data_dir = Path(data_dir)
    training_csv = data_dir / "train.csv"
    train_data = pd.read_csv(training_csv)
    
    for elem in tqdm(train_data.to_dict(orient='records'), "loading data"):
        path = elem["path"]
        label = elem["sign"]
        df = pd.read_parquet(data_dir / path)
        keypoints = get_keypoints(df)
        yield keypoints, label
        

    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)

    args = parser.parse_args()
    prepare_data(args.dir)