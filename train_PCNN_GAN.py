import torch
from torch.utils import data
import pandas as pd

from prep_data import PrepData
from loss import dis_loss, CalculateLoss
from model import Discriminator, PartialConvNet


if __name__ == "__main__":
    batch_size = 16
    lr = 0.01
    epochs = 10
    n_samples = 161000
    test_size = 1000
    j = 100
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    data_train = PrepData(n_samples=n_samples)
    print(f"Loaded training dataset with {data_train.num_imgs} samples")

    iters_per_epoch = (data_train.num_imgs - test_size) // batch_size

    generator = PartialConvNet().float()
    generator = torch.nn.DataParallel(generator)
    generator = generator.to(device)
    discriminator = Discriminator().float()
    discriminator = torch.nn.DataParallel(discriminator)
    discriminator = discriminator.to(device)
    print("Loaded model to device...")

    pytorch_total_params = sum(
        p.numel() for p in generator.parameters() if p.requires_grad
    ) + sum(
        p.numel() for p in discriminator.parameters() if p.requires_grad
    )
    print("Trainable model parameters:", pytorch_total_params)

    # möglicherweise filter nötig
    optimG = torch.optim.Adam(
        generator.parameters(),
        lr=lr,
    )

    optimD = torch.optim.Adam(
        discriminator.parameters(),
        lr=lr,
    )
    print("Setup Adam optimizer...")

    l1 = torch.nn.L1Loss()
    loss_weights = {"valid": 1.0, "hole": 10.0, "tv": 5.0, "perceptual": 0.1, "style": 50.0, "gen_loss": 0.2}
    gen_loss_func = CalculateLoss().to(device)
    dis_loss_func = dis_loss().to(device)
    print("Setup loss function...")

    loss_df = list()
    for epoch in range(1, epochs+1):
        iterator_train = iter(data.DataLoader(
            data_train,
            batch_size=batch_size,
        ))

        print(f"EPOCH:{epoch} of {epochs} - starting training loop from iteration:0 to iteration:{iters_per_epoch}")

        monitor_l1_loss = list()
        monitor_gen_loss = list()
        monitor_dis_loss = list()
        for i in range(1, iters_per_epoch+1):
            image, mask, gt = [x.float().to(device) for x in next(iterator_train)]

            pred_img = generator(image, mask)
            comp_img = (1 - mask) * gt + mask * pred_img

            loss_dict = gen_loss_func(loss_weights, image, mask, pred_img, gt, discriminator)
            dis_loss = dis_loss_func(discriminator, comp_img, gt)

            optimG.zero_grad()
            optimD.zero_grad()

            sum(loss_dict.values()).backward()
            dis_loss.backward()

            optimG.step()
            optimD.step()

            if i % j == 1:
                generator.eval()

                monitor_l1_loss.append(l1(comp_img, gt).item())
                monitor_gen_loss.append(loss_dict['gen_loss'].item())
                monitor_dis_loss.append(dis_loss.item())
                monitor_l1_loss = sum(monitor_l1_loss) / len(monitor_l1_loss)
                monitor_gen_loss = sum(monitor_gen_loss) / len(monitor_gen_loss)
                monitor_dis_loss = sum(monitor_dis_loss) / len(monitor_dis_loss)

                test_losses = list()
                with torch.no_grad():
                    for k in range(test_size):
                        test_image, test_mask, test_gt = [x.float().to(device) for x in data_train[data_train.num_imgs - test_size + k]]
                        test_image, test_mask, test_gt = test_image[None, :, :, :], test_mask[None, :, :, :], test_gt[None, :, :, :]

                        test_pred_img = generator(test_image, test_mask)

                        test_comp_img = (1 - test_mask) * test_gt + test_mask * test_pred_img
                        test_losses.append(l1(test_comp_img, test_gt).item())

                l1_loss = sum(test_losses) / len(test_losses)

                print(f"{i} l1: {round(monitor_l1_loss, 4)}, gen_los: {round(monitor_gen_loss, 4)}, dis_loss: {round(monitor_dis_loss, 4)}, l1_test: {round(l1_loss, 4)}")

                loss_df.append([epoch, i, monitor_l1_loss, monitor_gen_loss, monitor_dis_loss, l1_loss])

                monitor_l1_loss = list()
                monitor_gen_loss = list()
                monitor_dis_loss = list()

                generator.train()
            else:
                monitor_l1_loss.append(l1(comp_img, gt).item())
                monitor_gen_loss.append(loss_dict['gen_loss'].item())
                monitor_dis_loss.append(dis_loss.item())

    loss_df = pd.DataFrame(
        columns=['epoch', 'iteration', 'l1', 'generator_loss', 'discriminator_loss', 'l1_test'],
        data=loss_df
    )
    loss_df.to_csv(f"pcnn_gan_gen_lr_{lr}_epoch_{epochs}_batch_size_{batch_size}_nsamples_{n_samples}_test_size_{test_size}.csv", index=False, sep=';')

    torch.save(generator.state_dict(), f"pcnn_gan_gen_lr_{lr}_epoch_{epochs}_batch_size_{batch_size}_nsamples_{n_samples}_test_size_{test_size}.t7")
    torch.save(discriminator.state_dict(), f"pcnn_gan_dis_lr_{lr}_epoch_{epochs}_batch_size_{batch_size}_nsamples_{n_samples}_test_size_{test_size}.t7")
