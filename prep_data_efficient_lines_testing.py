import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.draw import line, disk
import os
import glob
import torch
from torchvision import transforms
#from tqdm import tqdm
import time


class PrepData(torch.utils.data.Dataset):
    def __init__(self, n_samples=3):
        super().__init__()

        self.n_samples = n_samples
        self.min_patch_size = 0.2
        self.max_patch_size = 0.3

        self.img_paths = glob.glob(os.path.dirname(os.path.abspath(__file__)) + '/data/data_celeba/*.jpg')[:self.n_samples]
        self.num_imgs = len(self.img_paths)

        self.img_transformer = transforms.ToTensor()
        
    def __getitem__(self, index):
        """
        Parameters to tweak:
        --- maxLines: Maximal number of random lines
        --- lines (low): Minimum number of random lines
        --- lowRad: Minimum radius of circles drawn on lines
        --- highRad: Maximum radius of circles drawn on lines
        --- function: Determines pattern of circle sizes on one line
        """
        img = Image.open(self.img_paths[index]).resize(size=(256, 256))
        img = self.img_transformer(img.convert('RGB'))
        # Determine how many lines should be defined
        # Can be tweaked:
        maxLines = 25
        lines = np.random.randint(1, maxLines)
        lines *= 2
        lowRad = 5
        highRad = 16

        # Init storing vectors
        all_line_rows = []
        all_line_cols = []
        all_circle_rows = []
        all_circle_cols = []
        
        # Mask init
        mask = torch.ones(size=img.shape, dtype=torch.float64)

        # Choose maximum radius of circles
        maxRad = np.random.randint(lowRad, highRad)
                
        # Generate x and y coordinates for lines: x[even]=x_start, x[odd]=x_end
        x = np.random.randint(1, img.shape[1]-1, size=lines)
        y = np.random.randint(1, img.shape[2]-1, size=lines)
        x = np.int_(x)
        y = np.int_(y)

        # Make lines
        for i in range(lines):
            if i % 2 == 0:
                row, col = line(x[i], y[i], x[i+1], y[i+1])
                # Store all line indices in these vectors
                all_line_rows = np.append(all_line_rows, row)
                all_line_cols = np.append(all_line_cols, col)

        # Decide how big the radius of disks should be        
        rand = np.random.randint(0, 1000, len(all_line_rows))

        # Draw circles for every line coordinate
        for i in range(len(all_line_rows)):
            # Draw only every 6th circle
            if i % 3 == 0:
                # Let radius vary
                function = maxRad*np.sin(rand[i])*2
                upperBound = min(max(function, lowRad), maxRad)
                radius = np.random.randint(lowRad-1, upperBound)

                rowCirc, colCirc = disk((all_line_rows[i], all_line_cols[i]), 
                                        radius, 
                                        shape=(img.shape[1], img.shape[2]))
                all_circle_rows = np.append(all_circle_rows, rowCirc)
                all_circle_cols = np.append(all_circle_cols, colCirc)

        # TODO: find a way to efficiently delete duplicates 
        # before this operation
        mask[:, all_circle_rows, all_circle_cols] = 0
        
        img = torch.as_tensor(img, dtype=torch.float64)
    
        return (img * mask), mask, img

if __name__ == '__main__':
    start = time.time()
    # Save masks as tensor files (.pt) and load them later to decrease learning time
    # for j in tqdm(range(1, 1001)):
    #     mi, m, i = PrepData()[1]
    #     torch.save(m, (os.getcwd() + f'\\masks\\mask_{j+1000}.pt'))
    mi, m, i = PrepData()[1]
    end = time.time()
    plt.imshow(mi.permute(1, 2, 0))
    plt.show()
    # print(mi.shape)
    # print(mi.dtype)
    # print(m.shape)
    # print(m.dtype)
    # print(i.shape)
    # print(i.dtype)
    print("Time of execution of the NEW version: ", end-start)

"""
Features lost due to optimization of speed:
 --- higher weighted probability for shorter lines
 --- 
"""