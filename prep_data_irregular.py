import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import skimage
import os
import glob
import torch
from torchvision import transforms


class PrepData(torch.utils.data.Dataset):
    def __init__(self, n_samples=100):
        super().__init__()

        self.n_samples = n_samples
        self.min_patch_size = 0.2
        self.max_patch_size = 0.3

        self.img_paths = glob.glob(os.path.dirname(os.path.abspath(__file__)) + '/data_celeba/*.jpg')[:self.n_samples]
        self.num_imgs = len(self.img_paths)

        self.img_transformer = transforms.ToTensor()
        
    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).resize(size=(256, 256))
        img = self.img_transformer(img.convert('RGB'))
        # Determine how many lines should be defined
        # Can be tweaked
        maxLines = 25
        lines = np.random.randint(1, maxLines)
        lines *= 2

        # Mask init
        mask = torch.ones(size=img.shape, dtype=torch.float64)
        # Make lines
        for i in range(lines):
            if i % 2 == 0:
                # Choose maximum radius randomly 
                maxRad = np.random.randint(6, 24)
                # Create vector of random numbers
                # Out of bounds errors may occur here
                x = np.random.randint(maxRad+1, img.shape[1]-maxRad-1, 2)
                y = np.random.randint(maxRad+1, img.shape[2]-maxRad-1, 2)
    
                row, col = skimage.draw.line(x[0], y[0], x[1], y[1])
                length = len(row)
                for j in range(length):
                    # TODO: find a better function that is more random but smooth
                    rand = np.random.randint(0, 10000)
                    # Hier können out of bounds-Fehler entstehen
                    function = maxRad*np.sin(rand)*2
                    upperBound = min(max(function, 5), maxRad)
                    radius = np.random.randint(4, upperBound)
                    rowCirc, colCirc = skimage.draw.disk((row[j], col[j]), radius)
                    mask[:, rowCirc, colCirc] = 0
    
        img = torch.as_tensor(img, dtype=torch.float64)
    
        return (img * mask), mask, img

# How to process multiple images?

if __name__ == '__main__':
    mi, m, i = PrepData()[1]
    plt.imshow(mi.permute(1, 2, 0))
    plt.show()
    print(mi.shape)
    print(mi.dtype)
    print(m.shape)
    print(m.dtype)
    print(i.shape)
    print(i.dtype)