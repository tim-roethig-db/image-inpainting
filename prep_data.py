import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def prep_data(n_samples, max_patch_size):
    for i in range(1, n_samples+1):
        img = Image.open(f"data_celeba/{i:06d}.jpg")
        img = np.array(img)

        mask = np.ones(shape=img.shape)

        y = np.random.randint(low=1 + int(0.05*img.shape[0]), high=img.shape[0])
        x = np.random.randint(low=1 + int(0.05*img.shape[1]), high=img.shape[1])

        range_y = int(np.random.randint(low=1, high=img.shape[0]) * max_patch_size / 2)
        range_x = int(np.random.randint(low=1, high=img.shape[1]) * max_patch_size / 2)

        patch_y_start = y - range_y
        patch_y_end = y + range_y
        patch_x_start = x - range_x
        patch_x_end = x + range_x

        mask[patch_y_start:patch_y_end, patch_x_start:patch_x_end, :] = 0

        img = (img * mask).astype(int)

        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    prep_data(1, 0.5)
