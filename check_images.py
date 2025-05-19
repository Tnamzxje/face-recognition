import os
import imageio.v2 as imageio
import numpy as np

root = "training-images"
for img_name in os.listdir(root):
    img_path = os.path.join(root, img_name)
    try:
        img = imageio.imread(img_path)
        if img.ndim == 2:
            img = np.stack((img,)*3, axis=-1)
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        img = np.ascontiguousarray(img)
        print(f"OK: {img_path} | dtype: {img.dtype}, shape: {img.shape}, contiguous: {img.flags['C_CONTIGUOUS']}")
    except Exception as e:
        print(f"LỖI: {img_path} | {e}")