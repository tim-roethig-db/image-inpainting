import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image


class PrepData(torch.utils.data.Dataset):
    def __init__(self, n_samples=1000):
        super().__init__()

        self.n_samples = n_samples
        self.min_patch_size = 0.2
        self.max_patch_size = 0.3

        self.img_paths = glob.glob('data/data_celeba/*.jpg')
        self.img_paths = self.img_paths[:self.n_samples]
        random.shuffle(self.img_paths)

        self.num_imgs = len(self.img_paths)

        self.img_transformer = transforms.ToTensor()

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).resize(size=(256, 256))
        img = self.img_transformer(img.convert('RGB'))

        y = np.random.randint(
            low=1 + int(self.max_patch_size * img.shape[1]),
            high=img.shape[1] - int(self.max_patch_size * img.shape[2]))
        x = np.random.randint(
            low=1 + int(self.max_patch_size * img.shape[1]),
            high=img.shape[2] - int(self.max_patch_size * img.shape[2]))

        range_y = int(np.random.randint(
            low=1 + int(self.min_patch_size * img.shape[1]),
            high=img.shape[1]) * self.max_patch_size / 2)
        range_x = int(np.random.randint(
            low=1 + int(self.min_patch_size * img.shape[2]),
            high=img.shape[2]) * self.max_patch_size / 2)

        patch_y_start = y - range_y
        patch_y_end = y + range_y
        patch_x_start = x - range_x
        patch_x_end = x + range_x

        mask = torch.ones(size=img.shape, dtype=torch.float64)
        mask[:, patch_y_start:patch_y_end, patch_x_start:patch_x_end] = 0

        img = torch.as_tensor(img, dtype=torch.float64)

        return (img * mask), mask, img

    def __len__(self):
        return self.n_samples


if __name__ == '__main__':
    mi, m, i = PrepData()[0]
    plt.imshow(mi.permute(1, 2, 0))
    plt.show()
    print(mi.shape)
    print(mi.dtype)
    print(m.shape)
    print(m.dtype)
    print(i.shape)
    print(i.dtype)
