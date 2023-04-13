import pandas as pd
from tqdm import tqdm
import numpy as np
from pathlib import Path

def to_npy(data_dir, output_dir):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    training_csv = data_dir / "train.csv"
    train_df = pd.read_csv(training_csv)
    for elem in tqdm(train_df.to_dict(orient='records'), "loading data"):
        path = data_dir / elem["path"]
        df = pd.read_parquet(path)
        
        
        arr = df[['x','y','z']].values
        n_frames = arr.shape[0] // 543
        arr = np.reshape(arr, (n_frames, 543 , 3))
        arr = arr.astype(np.float16)

        sequence_id = elem["sequence_id"]
        participant_id = elem["participant_id"]

        npy_path = output_dir / f"{participant_id}_{sequence_id}.npy"
        np.save(npy_path, arr)




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='Root dataset directory path', type=str, required=True)
    parser.add_argument('--output_dir', help='Root dataset directory path', type=str, required=True)

    args = parser.parse_args()
    to_npy(args.data_dir, args.output_dir)