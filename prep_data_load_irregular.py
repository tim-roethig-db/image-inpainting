import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import glob
import torch
from torchvision import transforms
from tqdm import tqdm


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
        img = torch.as_tensor(img, dtype=torch.float64)
        
        # Mask pt files have to be saved in image-inpainting/mask
        maskindex = np.random.randint(1, 2001)
        mask = torch.load(os.getcwd() + f'\\masks\\mask_{maskindex}.pt')
    
        return (img * mask), mask, img

if __name__ == '__main__':
    # Save masks as tensor files (.pt) and load them later to decrease learning time
    # for j in tqdm(range(1, 1001)):
    #     mi, m, i = PrepData()[1]
    #     torch.save(m, (os.getcwd() + f'\\masks\\mask_{j}.pt'))
    mi, m, i = PrepData()[1]
    plt.imshow(mi.permute(1, 2, 0))
    plt.show()
    # print(mi.shape)
    # print(mi.dtype)
    # print(m.shape)
    # print(m.dtype)
    # print(i.shape)
    # print(i.dtype)