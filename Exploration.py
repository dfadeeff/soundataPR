# ----------------------------
# Prepare training data from Metadata file
# ----------------------------
import pandas as pd
from pathlib import Path

download_path = Path('UrbanSound8K')

# Read metadata file
metadata_file = download_path / 'metadata' / 'UrbanSound8K.csv'
df = pd.read_csv(metadata_file)

# Construct file path by concatenating fold and file name
df['relative_path'] = '/fold' + df['fold'].astype(str) + '/' + df['slice_file_name'].astype(str)

# Take relevant columns
df = df[['relative_path', 'classID']]
if __name__ == "__main__":
    print(df.head())
