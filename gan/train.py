import torch
from torch.utils import data

from prep_data import PrepData
from loss import dis_loss, CalculateLoss
from model import InpaintGenerator, Discriminator


if __name__ == "__main__":
    batch_size = 12
    lr = 1e-4
    epochs = 2
    beta1 = 0.5
    beta2 = 0.999
    device = torch.device('cuda')

    data_train = PrepData(n_samples=batch_size * 100)
    print(f"Loaded training dataset with {data_train.num_imgs} samples")

    iters_per_epoch = data_train.num_imgs // batch_size

    generator = InpaintGenerator(rates=[1, 2, 4, 8], block_num=2).double().to(device)
    discriminator = Discriminator().double().to(device)
    print("Loaded model to device...")

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
    gen_loss_func = CalculateLoss().to(device)
    dis_loss_func = dis_loss().to(device)
    print("Setup loss function...")

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
            image, mask, gt = [x.to(device) for x in next(iterator_train)]
            mask = mask[:, 0, :, :]
            mask = mask[:, None, :, :]

            pred_img = generator(image, mask)
            comp_img = (1 - mask) * gt + mask * pred_img

            loss_dict = gen_loss_func(image, mask, pred_img, gt, discriminator)
            dis_loss = dis_loss_func(discriminator, comp_img, gt)

            optimG.zero_grad()
            optimD.zero_grad()
            sum(loss_dict.values()).backward()

            dis_loss.backward()
            optimG.step()
            optimD.step()

            j = 5
            if i % j == 0:
                monitor_l1_loss = monitor_l1_loss / j
                monitor_gen_loss = monitor_gen_loss / j
                monitor_dis_loss = monitor_dis_loss / j
                print(f"{i} l1: {round(monitor_l1_loss.item(), 4)}, gen_los: {round(monitor_gen_loss.item(), 4)}, dis_loss: {round(monitor_dis_loss.item(), 4)}")
                monitor_l1_loss = 0
                monitor_gen_loss = 0
                monitor_dis_loss = 0
            else:
                monitor_l1_loss += l1(comp_img, gt)
                monitor_gen_loss += loss_dict['gen_loss']
                monitor_dis_loss += dis_loss

    torch.save(generator.state_dict(), 'gan_generator')
    torch.save(discriminator.state_dict(), 'gan_discriminator')