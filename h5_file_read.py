import pandas as pd

meta = pd.read_csv("/home/adon/Downloads/Brats_datase/BraTS2020_training_data/content/data/meta_data.csv")
print(meta.head())

import h5py
import matplotlib.pyplot as plt

path = "/home/adon/Downloads/Brats_datase/BraTS2020_training_data/content/data/volume_100_slice_50.h5"

with h5py.File(path, 'r') as f:
    print("Keys:", list(f.keys()))
    for key in f.keys():
        print(f"{key}: shape={f[key].shape}, dtype={f[key].dtype}")

    # Visualize
    image = f['image'][:]

    mask = f['mask'][:]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image[:, :, 0], cmap='gray')  # first channel
plt.title("MRI Slice")
plt.subplot(1, 2, 2)
plt.imshow(mask, cmap='gray')
plt.title("Segmentation Mask")
plt.show()
