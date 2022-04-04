import torch
from torch.utils import data
import sys

#sys.path.insert(0, '..')

from prep_data import PrepData
from loss import L1, DLoss
from model import InpaintGenerator, Discriminator


if __name__ == "__main__":
    batch_size = 16
    lr = 1e-4
    epochs = 1
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

    l1 = L1()
    dloss = DLoss()
    print("Setup loss function...")

    for epoch in range(1, epochs+1):

        iterator_train = iter(data.DataLoader(
            data_train,
            batch_size=batch_size,
        ))

        # TRAINING LOOP
        print(f"EPOCH:{epoch} of {epochs} - starting training loop from iteration:0 to iteration:{iters_per_epoch}")

        for i in range(0, iters_per_epoch):
            # Gets the next batch of images
            image, mask, gt = [x.to(device) for x in next(iterator_train)]
            mask = mask[:, 0, :, :]
            mask = mask[:, None, :, :]

            pred_img = generator(image, mask)
            comp_img = (1 - mask) * gt + mask * pred_img

            losses = {}
            losses['l1'] = l1(pred_img, gt)
            dis_loss, gen_loss = dloss(discriminator, comp_img, gt)
            losses['gen_loss'] = gen_loss

            optimG.zero_grad()
            optimD.zero_grad()
            sum(losses.values()).backward()
            # gen_loss.backward()
            losses['dis_loss'] = dis_loss
            dis_loss.backward()
            optimG.step()
            optimD.step()

            # logs
            #if (i + 1) % 100 == 0:
            #    print(i + 1, ':', losses)
            print(i, ': ', losses)

    torch.save(generator.state_dict(), 'gan_generator')
    torch.save(discriminator.state_dict(), 'gan_discriminator')