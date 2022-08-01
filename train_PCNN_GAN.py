import torch
from torch.utils import data
import pandas as pd

from prep_data import PrepData
from loss import dis_loss, CalculateLoss
from model import InpaintGenerator, Discriminator, PartialConvNet


if __name__ == "__main__":
    batch_size = 2
    lr = 0.01 #0.01 für PCNN GAN
    epochs = 2
    beta1 = 0.5
    beta2 = 0.999
    n_samples = 10
    test_size = 2
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    data_train = PrepData(n_samples=n_samples)
    print(f"Loaded training dataset with {data_train.num_imgs} samples")

    iters_per_epoch = (data_train.num_imgs - test_size) // batch_size

    generator = PartialConvNet().double()
    generator = torch.nn.DataParallel(generator)
    generator = generator.to(device)
    discriminator = Discriminator().double()
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
        betas=(beta1, beta2)
    )

    optimD = torch.optim.Adam(
        discriminator.parameters(),
        lr=lr,
        betas=(beta1, beta2)
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

        # TRAINING LOOP
        print(f"EPOCH:{epoch} of {epochs} - starting training loop from iteration:0 to iteration:{iters_per_epoch}")

        monitor_l1_loss = 0
        monitor_gen_loss = 0
        monitor_dis_loss = 0
        for i in range(1, iters_per_epoch+1):
            # Gets the next batch of images
            image, mask, gt = [x.double().to(device) for x in next(iterator_train)]

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

            j = 1
            if i % j == 0:
                monitor_l1_loss += l1(comp_img, gt)
                monitor_gen_loss += loss_dict['gen_loss']
                monitor_dis_loss += dis_loss
                monitor_l1_loss = monitor_l1_loss / j
                monitor_gen_loss = monitor_gen_loss / j
                monitor_dis_loss = monitor_dis_loss / j

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

                generator.eval()
                with torch.no_grad():
                    pred_img = generator(image, mask)
                generator.train()

                comp_img = (1 - mask) * ground_truth + mask * pred_img
                l1_loss = l1(comp_img, ground_truth).item()
                print(f"{i} l1: {round(monitor_l1_loss.item(), 4)}, gen_los: {round(monitor_gen_loss.item(), 4)}, dis_loss: {round(monitor_dis_loss.item(), 4)}, l1_test: {round(l1_loss, 4)}")

                loss_df.append([epoch, i, monitor_l1_loss.item(), monitor_gen_loss.item(), monitor_dis_loss.item(), l1_loss])

                monitor_l1_loss = 0
                monitor_gen_loss = 0
                monitor_dis_loss = 0
            else:
                monitor_l1_loss += l1(comp_img, gt)
                monitor_gen_loss += loss_dict['gen_loss']
                monitor_dis_loss += dis_loss

    loss_df = pd.DataFrame(
        columns=['epoch', 'iteration', 'l1', 'generator_loss', 'discriminator_loss', 'l1_test'],
        data=loss_df
    )
    loss_df.to_csv(f"pcnn_gan_gen_lr_{lr}_epoch_{epochs}_batch_size_{batch_size}.csv", index=False, sep=';')

    torch.save(generator.state_dict(), f"gan_gen_lr_{lr}_epoch_{epochs}_batch_size_{batch_size}.t7")
    torch.save(discriminator.state_dict(), f"gan_dis_lr_{lr}_epoch_{epochs}_batch_size_{batch_size}.t7")
