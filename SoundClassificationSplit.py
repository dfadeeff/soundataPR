# SoundClassificationSplit.py
from pathlib import Path
import pandas as pd
from SoundsDS import SoundDS
from torch.utils.data import random_split, DataLoader

def get_data_loaders(batch_size=16, split_ratio=0.8):
    download_path = Path('UrbanSound8K')
    metadata_file = download_path / 'metadata' / 'UrbanSound8K.csv'

    df = pd.read_csv(metadata_file)
    df['relative_path'] = '/fold' + df['fold'].astype(str) + '/' + df['slice_file_name'].astype(str)
    df = df[['relative_path', 'classID']]

    data_path = download_path / 'audio'
    dataset = SoundDS(df, data_path)

    # Split into train/val
    total = len(dataset)
    train_size = round(total * split_ratio)
    val_size = total - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_dl, val_dl