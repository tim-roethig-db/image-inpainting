import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.draw import line, disk
import os
import glob
import torch
from torchvision import transforms
#from tqdm import tqdm


class PrepData(torch.utils.data.Dataset):
    def __init__(self, n_samples):
        super().__init__()

        self.n_samples = n_samples
        self.min_patch_size = 0.2
        self.max_patch_size = 0.3

        id = os.environ["SLURM_JOB_ID"]
        #self.img_paths = glob.glob(os.path.dirname(os.path.abspath(__file__)) + '/data/data_celeba/*.jpg')[:self.n_samples]
        self.img_paths = glob.glob(f'/scratch/{id}' + '/data/data_celeba/*.jpg')[:self.n_samples]
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
        lowRad = 3
        highRad = 16
        
        ps = np.linspace(0.95, 0, num=200, endpoint=False)
        ps = ps / ps.sum()
        linebounds = np.arange(1, 201)
        
        # Mask init
        mask = torch.ones(size=img.shape, dtype=torch.float64)
        # Make lines
        for i in range(lines):
            if i % 2 == 0:
                # Choose maximum radius randomly
                maxRad = np.random.randint(lowRad, highRad)
                # Create vector of random numbers
                # Out of bounds errors may occur here
                
                # Uncomment to increase probability of short lines for more patchy look
                # Also uncomment linebounds and ps above for loop
                x = np.zeros(2)
                y = np.zeros(2)
                linebound = np.random.choice(linebounds, p=ps)
                x[0] = np.random.randint(maxRad+1, img.shape[1]-maxRad-1)
                x[1] = np.random.randint(max(maxRad+1, x[0]-linebound), min(img.shape[1]-maxRad-1, x[0]+linebound))
                y[0] = np.random.randint(maxRad+1, img.shape[2]-maxRad-1)
                y[1] = np.random.randint(max(maxRad+1, y[0]-linebound), min(img.shape[2]-maxRad-1, y[0]+linebound))
                x = np.int_(x)
                y = np.int_(y)

                # # TODO: Now: all lines are more in the center, since maxRad is used as bound for all lines, but they don´t have circles with maxRad
                # x = np.random.randint(maxRad+1, img.shape[1]-maxRad-1, 2)
                # y = np.random.randint(maxRad+1, img.shape[2]-maxRad-1, 2)
    
                row, col = line(x[0], y[0], x[1], y[1])
                length = len(row)
                # Draw arbitrary circles on each point of each line
                for j in range(length):
                    # TODO: find a better function that is more random and smoother
                    rand = np.random.randint(0, 10000)
                    # Hier können out of bounds-Fehler entstehen
                    function = maxRad*np.sin(rand)*2
                    upperBound = min(max(function, lowRad), maxRad)
                    radius = np.random.randint(lowRad-1, upperBound)
                    rowCirc, colCirc = disk((row[j], col[j]), radius)
                    mask[:, rowCirc, colCirc] = 0
    
        img = torch.as_tensor(img, dtype=torch.float64)
    
        return (img * mask), mask, img

if __name__ == '__main__':
    # Save masks as tensor files (.pt) and load them later to decrease learning time
    # for j in tqdm(range(1, 1001)):
    #     mi, m, i = PrepData()[1]
    #     torch.save(m, (os.getcwd() + f'\\masks\\mask_{j+1000}.pt'))
    mi, m, i = PrepData()[1]
    plt.imshow(mi.permute(1, 2, 0))
    plt.show()
    # print(mi.shape)
    # print(mi.dtype)
    # print(m.shape)
    # print(m.dtype)
    # print(i.shape)
    # print(i.dtype)