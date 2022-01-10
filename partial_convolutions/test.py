import matplotlib.pyplot as plt
import torch

from model import PartialConvNet
from prep_data import PrepData

device = torch.device('cpu')

model = PartialConvNet().double()
model.load_state_dict(torch.load('pc_model'))
model = model.to(device)
model.eval()

img, mask, gt_img = PrepData()[0]

plt.imshow(img.permute(1, 2, 0))
plt.show()

img.unsqueeze_(0)
gt_img.unsqueeze_(0)
mask.unsqueeze_(0)
print(img)

with torch.no_grad():
    output = model(img.to(device), mask.to(device))
print(output)
#output = (mask * img) + ((1 - mask) * output)
#print(output)

plt.imshow(output[0].permute(1, 2, 0))
plt.show()