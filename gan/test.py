import random

import matplotlib.pyplot as plt
import torch

from model import InpaintGenerator, PartialConvNet
from prep_data import PrepData

if __name__ == '__main__':
    device = torch.device('cpu')

    # model = InpaintGenerator([1, 2, 4, 8], 2).double()
    model = PartialConvNet().double()
    model.load_state_dict(torch.load('gan_partial', map_location=torch.device('cpu')))
    model = model.to(device)
    model.eval()

    i = random.randint(0, 1000)
    img, mask, gt_img = PrepData()[i]

    plt.imshow(img.permute(1, 2, 0))
    plt.show()

    img.unsqueeze_(0)
    gt_img.unsqueeze_(0)
    mask.unsqueeze_(0)

    with torch.no_grad():
        output = model(img.to(device), mask.to(device))
    mask.to(device)
    img.to(device)
    output.to(device)
    output = (mask * img) + ((1 - mask) * output)

    plt.imshow(output[0].permute(1, 2, 0))
    plt.show()
