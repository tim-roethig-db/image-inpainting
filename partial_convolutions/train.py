import torch
from torch.utils import data
import pandas as pd

from prep_data import PrepData
from model import PartialConvNet
from loss import CalculateLoss


def requires_grad(param):
    return param.requires_grad


if __name__ == '__main__':
    batch_size = 2
    lr = 0.01
    epochs = 1
    device = torch.device('cpu')

    data_train = PrepData(n_samples=batch_size * 2)
    print(f"Loaded training dataset with {data_train.num_imgs} samples")

    iters_per_epoch = data_train.num_imgs // batch_size

    model = PartialConvNet().double().to(device)
    print("Loaded model to device...")

    optimizer = torch.optim.Adam(filter(requires_grad, model.parameters()), lr=lr)
    print("Setup Adam optimizer...")

    l1 = torch.nn.L1Loss()
    loss_func = CalculateLoss().to(device)
    print("Setup loss function...")

    loss_df = list()
    for epoch in range(1, epochs + 1):
        iterator_train = iter(data.DataLoader(
            data_train,
            batch_size=batch_size, ))

        # TRAINING LOOP
        print(f"EPOCH:{epoch} of {epochs} - starting training loop from iteration:0 to iteration:{iters_per_epoch}")

        monitor_l1_loss = 0
        for i in range(1, iters_per_epoch + 1):
            # Sets model to train mode
            #model.train()

            # Gets the next batch of images
            image, mask, gt = [x.to(device) for x in next(iterator_train)]

            # Forward-propagates images through net
            # Mask is also propagated, though it is usually gone by the decoding stage
            pred_img = model(image, mask)
            comp_img = (1 - mask) * gt + mask * pred_img

            loss_dict = loss_func(image, mask, pred_img, gt)

            # Resets gradient accumulator in optimizer
            optimizer.zero_grad()
            # back-propogates gradients through model weights
            sum(loss_dict.values()).backward()
            # updates the weights
            optimizer.step()

            j = 1
            if i % j == 0:
                monitor_l1_loss += l1(comp_img, gt)
                monitor_l1_loss = monitor_l1_loss / j
                print(f"{i} l1: {round(monitor_l1_loss.item(), 4)}")
                loss_df.append([epoch, i, monitor_l1_loss.item()])
                monitor_l1_loss = 0
            else:
                monitor_l1_loss += l1(comp_img, gt)

        loss_df = pd.DataFrame(
            columns=['epoch', 'iteration', 'l1'],
            data=loss_df
        )
        loss_df.to_csv('losses.csv', index=False)

    torch.save(model.state_dict(), 'pc_model')
