import torch
import torch.nn as nn


class L1:
    def __init__(self):
        self.calc = nn.L1Loss()

    def __call__(self, x, y):
        return self.calc(x, y)


class DLoss:
    def __init__(self, ):
        self.loss_fn = torch.nn.Softplus()

    def __call__(self, netD, fake, real):
        fake_detach = fake.detach()
        d_fake = netD(fake_detach)
        d_real = netD(real)
        dis_loss = self.loss_fn(-d_real).mean() + self.loss_fn(d_fake).mean()

        g_fake = netD(fake)
        gen_loss = self.loss_fn(-g_fake).mean()

        return dis_loss, gen_loss
