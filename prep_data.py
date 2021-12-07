import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def prep_data(n_samples):
    for i in range(1, n_samples+1):
        img = Image.open(f"data_celeba/{i:06d}.jpg")
        img = np.array(img)
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    prep_data(1)
