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
    n_samples = 10
    test_size = 2
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    data_train = PrepData(n_samples=n_samples)
    print(f"Loaded training dataset with {data_train.num_imgs} samples")

    iters_per_epoch = (data_train.num_imgs - test_size) // batch_size

    model = PartialConvNet().double()
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    print("Loaded model to device...")

    pytorch_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print("Trainable model parameters:", pytorch_total_params)

    optimizer = torch.optim.Adam(filter(requires_grad, model.parameters()), lr=lr)
    print("Setup Adam optimizer...")

    l1 = torch.nn.L1Loss()
    loss_weights = {"valid": 1.0, "hole": 10.0, "tv": 5.0, "perceptual": 0.1, "style": 50.0, "gen_loss": None}
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

            loss_dict = loss_func(loss_weights, image, mask, pred_img, gt)

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

                image, mask, ground_truth = [], [], []
                for k in range(test_size):
                    im, m, gt = [x.to(device) for x in data_train[data_train.num_imgs - test_size + k]]
                    im, m, gt = im[None, :, :, :], m[None, :, :, :], gt[None, :, :, :]
                    image.append(im)
                    mask.append(m)
                    ground_truth.append(gt)

                image = torch.cat(image)
                mask = torch.cat(mask)
                ground_truth = torch.cat(ground_truth)

                model.eval()
                with torch.no_grad():
                    pred_img = model(image, mask)
                model.train()

                comp_img = (1 - mask) * ground_truth + mask * pred_img
                l1_loss = l1(comp_img, ground_truth).item()
                loss_df.append([epoch, i, monitor_l1_loss.item(), l1_loss])

                monitor_l1_loss = 0
            else:
                monitor_l1_loss += l1(comp_img, gt)

    loss_df = pd.DataFrame(
        columns=['epoch', 'iteration', 'l1', 'l1_test'],
        data=loss_df
    )
    loss_df.to_csv(f"pcnn_lr_{lr}_epoch_{epochs}_batch_size_{batch_size}.csv", index=False, sep=';')

    torch.save(model.state_dict(), f"pcnn_lr_{lr}_epoch_{epochs}_batch_size_{batch_size}.t7")

