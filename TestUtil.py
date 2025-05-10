import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from AudioUtil import AudioUtil  # import your utility

# Step 1: Load metadata
download_path = Path('UrbanSound8K')
metadata_file = download_path / 'metadata' / 'UrbanSound8K.csv'
df = pd.read_csv(metadata_file)
df['relative_path'] = '/fold' + df['fold'].astype(str) + '/' + df['slice_file_name'].astype(str)

# Step 2: Pick an audio file to test
sample_path = download_path / 'audio' / df.iloc[0]['relative_path'].lstrip('/')
print(f"Loading: {sample_path}")

# Step 3: Process the audio
aud = AudioUtil.open(sample_path)
aud = AudioUtil.rechannel(aud, 1)
aud = AudioUtil.resample(aud, 44100)
aud = AudioUtil.pad_trunc(aud, 4000)
aud = AudioUtil.time_shift(aud, 0.2)

# Step 4: Generate spectrogram
spec = AudioUtil.spectro_gram(aud)

# Step 5: Augment spectrogram
aug_spec = AudioUtil.spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)


if __name__ == "__main__":
    # Step 6: Visualize original and augmented spectrogram
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.title("Original Spectrogram")
    plt.imshow(spec[0].numpy(), origin='lower', aspect='auto')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("Augmented Spectrogram")
    plt.imshow(aug_spec[0].numpy(), origin='lower', aspect='auto')
    plt.colorbar()

    plt.tight_layout()
    plt.show()