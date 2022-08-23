import torch
from torch.utils import data
import pandas as pd

from prep_data import PrepData
from model import PartialConvNet
from loss import CalculateLoss


if __name__ == '__main__':
    batch_size = 16
    lr = 0.01
    epochs = 1
    n_samples = 4216
    test_size = 1000
    j = 100
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    data_train = PrepData(n_samples=n_samples)
    print(f"Loaded training dataset with {data_train.num_imgs} samples")

    iters_per_epoch = (data_train.num_imgs - test_size) // batch_size

    model = PartialConvNet().float()
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    print("Loaded model to device...")

    pytorch_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print("Trainable model parameters:", pytorch_total_params)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )
    print("Setup Adam optimizer...")

    l1 = torch.nn.L1Loss()
    loss_weights = {"valid": 1.0, "hole": 10.0, "tv": 5.0, "perceptual": 0.1, "style": 50.0, "gen_loss": None}
    loss_func = CalculateLoss().to(device)
    print("Setup loss function...")

    loss_df = list()
    model.train()
    for epoch in range(1, epochs + 1):
        iterator_train = iter(data.DataLoader(
            data_train,
            batch_size=batch_size,
        ))

        print(f"EPOCH:{epoch} of {epochs} - starting training loop from iteration:0 to iteration:{iters_per_epoch}")

        monitor_l1_loss = list()
        for i in range(1, iters_per_epoch + 1):
            image, mask, gt = [x.float().to(device) for x in next(iterator_train)]

            pred_img = model(image, mask)
            comp_img = (1 - mask) * gt + mask * pred_img

            loss_dict = loss_func(loss_weights, image, mask, pred_img, gt)

            optimizer.zero_grad()

            sum(loss_dict.values()).backward()

            optimizer.step()

            if i % j == 1:
                model.eval()

                monitor_l1_loss.append(l1(comp_img, gt).item())
                monitor_l1_loss = sum(monitor_l1_loss) / len(monitor_l1_loss)

                test_losses = list()
                with torch.no_grad():
                    for k in range(test_size):
                        test_image, test_mask, test_gt = [x.float().to(device) for x in data_train[data_train.num_imgs - test_size + k]]
                        test_image, test_mask, test_gt = test_image[None, :, :, :], test_mask[None, :, :, :], test_gt[None, :, :, :]

                        test_pred_img = model(test_image, test_mask)

                        test_comp_img = (1 - test_mask) * test_gt + test_mask * test_pred_img
                        test_losses.append(l1(test_comp_img, test_gt).item())

                l1_loss = sum(test_losses) / len(test_losses)

                print(f"{i} l1: {round(monitor_l1_loss, 4)}, l1_test: {round(l1_loss, 4)}")
                loss_df.append([epoch, i, monitor_l1_loss, l1_loss])

                monitor_l1_loss = list()

                model.train()
            else:
                monitor_l1_loss.append(l1(comp_img, gt).item())

    loss_df = pd.DataFrame(
        columns=['epoch', 'iteration', 'l1', 'l1_test'],
        data=loss_df
    )
    loss_df.to_csv(f"pcnn_lr_{lr}_epoch_{epochs}_batch_size_{batch_size}_nsamples_{n_samples}_test_size_{test_size}.csv", index=False, sep=';')

    torch.save(model.state_dict(), f"pcnn_lr_{lr}_epoch_{epochs}_batch_size_{batch_size}_nsamples_{n_samples}_test_size_{test_size}.t7")

