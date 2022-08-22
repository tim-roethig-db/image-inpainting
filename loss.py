import torch
import torch.nn as nn
from torchvision import models

LAMBDAS = {"valid": 1.0, "hole": 1.0, "tv": 1.0, "perceptual": 1.0, "style": 1.0, "gen_loss": 1.0}


def gram_matrix(feature_matrix):
    (batch, channel, h, w) = feature_matrix.size()
    feature_matrix = feature_matrix.view(batch, channel, h * w)
    feature_matrix_t = feature_matrix.transpose(1, 2)

    # batch matrix multiplication * normalization factor K_n
    # (batch, channel, h * w) x (batch, h * w, channel) ==> (batch, channel, channel)
    gram = torch.bmm(feature_matrix, feature_matrix_t) / (channel * h * w)

    # size = (batch, channel, channel)
    return gram


def perceptual_loss(h_comp, h_out, h_gt, l1):
    loss = 0.0

    for i in range(len(h_comp)):
        loss += l1(h_out[i], h_gt[i])
        loss += l1(h_comp[i], h_gt[i])

    return loss


def style_loss(h_comp, h_out, h_gt, l1):
    loss = 0.0

    for i in range(len(h_comp)):
        loss += l1(gram_matrix(h_out[i]), gram_matrix(h_gt[i]))
        loss += l1(gram_matrix(h_comp[i]), gram_matrix(h_gt[i]))

    return loss


# computes TV loss over entire composed image since gradient will not be passed backward to input
def total_variation_loss(image, l1):
    # shift one pixel and get loss1 difference (for both x and y direction)
    loss = l1(image[:, :, :, :-1], image[:, :, :, 1:]) + l1(image[:, :, :-1, :], image[:, :, 1:, :])

    return loss


def gen_loss(netD, fake):
    loss_fn = torch.nn.Softplus()

    g_fake = netD(fake)
    gen_loss = loss_fn(-g_fake).mean()

    return gen_loss


class VGG16Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True).float()
        self.max_pooling1 = vgg16.features[:5]
        self.max_pooling2 = vgg16.features[5:10]
        self.max_pooling3 = vgg16.features[10:17]

        for i in range(1, 4):
            for param in getattr(self, 'max_pooling{:d}'.format(i)).parameters():
                param.requires_grad = False

    # feature extractor at each of the first three pooling layers
    def forward(self, image):
        results = [image]
        for i in range(1, 4):
            func = getattr(self, 'max_pooling{:d}'.format(i))
            results.append(func(results[-1]))
        return results[1:]


class CalculateLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg_extract = VGG16Extractor()
        self.l1 = nn.L1Loss()

    def forward(self, weights, input_x, mask, output, ground_truth, netD=None):
        composed_output = (input_x * mask) + (output * (1 - mask))
        print(composed_output.dtype)
        fs_composed_output = self.vgg_extract(composed_output)
        fs_output = self.vgg_extract(output)
        fs_ground_truth = self.vgg_extract(ground_truth)

        loss_dict = dict()

        if weights["hole"] is not None:
            loss_dict["hole"] = self.l1((1 - mask) * output, (1 - mask) * ground_truth) * weights["hole"]
        if weights["valid"] is not None:
            loss_dict["valid"] = self.l1(mask * output, mask * ground_truth) * weights["valid"]
        if weights["perceptual"] is not None:
            loss_dict["perceptual"] = perceptual_loss(fs_composed_output, fs_output, fs_ground_truth, self.l1) * weights["perceptual"]
        if weights["style"] is not None:
            loss_dict["style"] = style_loss(fs_composed_output, fs_output, fs_ground_truth, self.l1) * weights["style"]
        if weights["tv"] is not None:
            loss_dict["tv"] = total_variation_loss(composed_output, self.l1) * weights["tv"]
        if weights["gen_loss"] is not None:
            loss_dict["gen_loss"] = gen_loss(netD, composed_output) * weights['gen_loss']

        return loss_dict


class dis_loss(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.loss_fn = torch.nn.Softplus()

    def forward(self, netD, fake, real):
        fake_detach = fake.detach()
        d_fake = netD(fake_detach)
        d_real = netD(real)
        dis_loss = self.loss_fn(-d_real).mean() + self.loss_fn(d_fake).mean()

        return dis_loss
