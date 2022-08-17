import random

import matplotlib.pyplot as plt
import torch
import pandas as pd

from model import InpaintGenerator, PartialConvNet, PartialConvNetNew
from prep_data import PrepData

if __name__ == '__main__':
    device = torch.device('cuda')

    model_gan = InpaintGenerator([1, 2, 4, 8], 2).double()
    model_gan = torch.nn.DataParallel(model_gan)
    model_gan = model_gan.to(device)

    model_pcnn = PartialConvNet().double()
    # model_pcnn = torch.nn.DataParallel(model_pcnn)
    # model_pcnn = model_pcnn.to(device)

    model_pcnn_gan = PartialConvNet().double()
    # model_pcnn_gan = torch.nn.DataParallel(model_pcnn_gan)
    # model_pcnn_gan = model_pcnn_gan.to(device)

    model_pcnn_new = PartialConvNetNew().double()
    model_pcnn_new = torch.nn.DataParallel(model_pcnn_new)
    model_pcnn_new = model_pcnn_new.to(device)

    model_gan.load_state_dict(torch.load('mein_gan_run_gen.t7', map_location=torch.device('cuda')))
    model_pcnn.load_state_dict(torch.load('../partial_convolutions/pc_model.t7', map_location=torch.device('cuda')))
    model_pcnn_gan.load_state_dict(torch.load('pcnn_gan_generator.t7', map_location=torch.device('cuda')))
    model_pcnn_new.load_state_dict(torch.load('../partial_convolutions/new_pc.t7', map_location=torch.device('cuda')))

    model_gan = model_gan.to(device)
    model_pcnn = model_pcnn.to(device)
    model_pcnn_gan = model_pcnn_gan.to(device)
    model_pcnn_new = model_pcnn_new.to(device)

    model_gan.eval()
    model_pcnn.eval()
    model_pcnn_gan.eval()
    model_pcnn_new.eval()

    l1 = torch.nn.L1Loss()

    l1_losses = {'gan': [], 'pcnn': [], 'pcnn_gan': [], 'pcnn_new': []}
    n = 10
    for i in range(n):
        img, mask, gt_img = PrepData()[i]

        img.unsqueeze_(0)
        gt_img.unsqueeze_(0)
        mask.unsqueeze_(0)
        mask = mask.to(device)
        img = img.to(device)
        gt_img = gt_img.to(device)
        with torch.no_grad():
            output_gan = model_gan(img, mask)
            output_pcnn = model_pcnn(img, mask)
            output_pcnn_gan = model_pcnn_gan(img, mask)

        output_gan = (mask * img) + ((1 - mask) * output_gan)
        output_pcnn = (mask * img) + ((1 - mask) * output_pcnn)
        output_pcnn_gan = (mask * img) + ((1 - mask) * output_pcnn_gan)

        l1_losses['gan'].append(l1(output_gan, gt_img))
        l1_losses['pcnn'].append(l1(output_pcnn, gt_img))
        l1_losses['pcnn_gan'].append(l1(output_pcnn_gan, gt_img))

    df = pd.DataFrame.from_dict(l1_losses)
    df.to_csv('evaluation_losses.csv', index=False)
