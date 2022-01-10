import torch
import torch.nn as nn
from torch.utils import data

from prep_data import PrepData
from model import PartialConvNet
from loss import CalculateLoss


class SubsetSampler(data.sampler.Sampler):
    def __init__(self, start_sample, num_samples):
        self.num_samples = num_samples
        self.start_sample = start_sample

    def __iter__(self):
        return iter(range(self.start_sample, self.num_samples))

    def __len__(self):
        return self.num_samples


def requires_grad(param):
    return param.requires_grad


if __name__ == '__main__':
    batch_size = 8
    lr = 0.1
    epochs = 2
    num_workers = 1
    device = torch.device('cpu')

    data_train = PrepData(n_samples=batch_size * 10)
    print(f"Loaded training dataset with {data_train.num_imgs} samples")

    iters_per_epoch = data_train.num_imgs // batch_size

    model = PartialConvNet().double().to(device)
    print("Loaded model to device...")

    optimizer = torch.optim.Adam(filter(requires_grad, model.parameters()), lr=lr)
    print("Setup Adam optimizer...")

    loss_func = CalculateLoss().to(device)
    print("Setup loss function...")

    for epoch in range(1, epochs+1):

        iterator_train = iter(data.DataLoader(
            data_train,
            batch_size=batch_size,
            # num_workers=num_workers,
            sampler=SubsetSampler(0, data_train.num_imgs)))

        # TRAINING LOOP
        print(f"EPOCH:{epoch} of {epochs} - starting training loop from iteration:0 to iteration:{iters_per_epoch}")

        for i in range(0, iters_per_epoch):
            # Sets model to train mode
            model.train()

            # Gets the next batch of images
            image, mask, gt = [x.to(device) for x in next(iterator_train)]

            # Forward-propagates images through net
            # Mask is also propagated, though it is usually gone by the decoding stage
            output = model(image, mask)

            loss_dict = loss_func(mask, output, gt)
            loss = 0.0
            # sums up each loss value
            for key, value in loss_dict.items():
                loss += value
            print(loss)

            # Resets gradient accumulator in optimizer
            optimizer.zero_grad()
            # back-propogates gradients through model weights
            loss.backward()
            # updates the weights
            optimizer.step()

    torch.save(model.state_dict(), 'pc_model')
