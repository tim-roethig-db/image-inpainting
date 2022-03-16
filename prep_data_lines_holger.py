import numpy as np
from skimage import draw
import torch
from matplotlib import pyplot as plt
#%matplotlib inline
from PIL import Image
import os
import glob
import torch
from torchvision import transforms

def get_gauss_filter(sigma=0, k=1):
    """Returns a gaussian filter mask."""
    if sigma == 0:
        mask = torch.ones(shape=(1,1), dtype=torch.float32)
    else:
        gauss = lambda x: torch.exp(-0.5 * x / sigma**2)
        w = 2 * k + 1
        coords = (torch.arange(w).view(w,1) - k)**2
        mask = gauss(coords + coords.T)
        mask = mask.to(dtype=torch.float32)
        mask = mask / mask[k,k]
    return mask

class PrepData(torch.utils.data.Dataset):
    def __init__(self, n_samples=100):
        super().__init__()

        self.n_samples = n_samples
        self.min_patch_size = 0.2
        self.max_patch_size = 0.3

        # This part is important for running on cluster
        #id = os.environ["SLURM_JOB_ID"]
        self.img_paths = glob.glob(os.path.dirname(os.path.abspath(__file__)) + '/data/data_celeba/*.jpg')[:self.n_samples]
        #self.img_paths = glob.glob(f'/scratch/{id}' + '/data/data_celeba/*.jpg')[:self.n_samples]
        self.num_imgs = len(self.img_paths)

        self.img_transformer = transforms.ToTensor()
        
    def __getitem__(self, index):
        
        img = Image.open(self.img_paths[index]).resize(size=(256, 256))
        img = self.img_transformer(img.convert('RGB'))

        # choose parameters
        batchsize = 2
        w, h = 256, 256
        num_lines = np.random.randint(2,16)
        # relation of num_lines and sigma can be improved
        # thinner lines the more lines are present
        sigma = int(8/num_lines + 1)
        k = 2 * sigma
        tau = 1

        # elegant way of drawing multiple lines
        xs = np.random.randint(0,w, size=(num_lines, 2))
        ys = np.random.randint(0,h, size=(num_lines, 2))
        rowscols = [torch.tensor(draw.line(x0, y0, x1, y1)) for (x0, x1), (y0, y1) in zip(xs, ys)]

        rows, cols = torch.cat(rowscols, dim=1)
        num_points, = rows.shape

        # Create empty mask
        mask = torch.zeros(h,w, dtype=torch.float32)

        # Fill mask at line points with random positive values
        mask[rows, cols] = (4*torch.randn(num_points))**2 + 1

        gauss_filter = get_gauss_filter(sigma=sigma, k=k)

        # Blurr the mask using the previously created gauss filter.
        mask = torch.conv2d(mask.view(1,1,h,w), gauss_filter[None,None], padding=k)[0,0]

        # Apply a threshold tau to obtain a binary mask defining the irregular line.
        mask = (mask >= tau)
        # Invert boolean mask to receive final mask overlay
        mask = ~mask

        img = torch.as_tensor(img, dtype=torch.float64)
    
        return (img * mask), mask, img

if __name__ == '__main__':
    mi, m, i = PrepData()[1]
    plt.imshow(mi.permute(1, 2, 0))
    plt.show()
