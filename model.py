import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        try:
            m.bias.data.fill_(0.01)
        except:
            pass


def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat

    return feat


class InpaintGenerator(nn.Module):
    def __init__(self, rates, block_num):
        super(InpaintGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(True)
        )
        self.encoder.apply(init_weights)

        self.middle = nn.Sequential(*[AOTBlock(256, rates) for _ in range(block_num)])
        self.middle.apply(init_weights)

        self.decoder = nn.Sequential(
            UpConv(256, 128),
            nn.ReLU(True),
            UpConv(128, 64),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1)
        )
        self.decoder.apply(init_weights)

    def forward(self, x, mask):
        x = x * mask
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.tanh(x)
        return x


class UpConv(nn.Module):
    def __init__(self, inc, outc, scale=2):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv = nn.Conv2d(inc, outc, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True))


class AOTBlock(nn.Module):
    def __init__(self, dim, rates):
        super(AOTBlock, self).__init__()
        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                'block{}'.format(str(i).zfill(2)),
                nn.Sequential(
                    nn.ReflectionPad2d(rate),
                    nn.Conv2d(dim, dim // 4, 3, padding=0, dilation=rate),
                    nn.ReLU(True)))
        self.fuse = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1))
        self.gate = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1))

    def forward(self, x):
        out = [self.__getattr__(f'block{str(i).zfill(2)}')(x) for i in range(len(self.rates))]
        out = torch.cat(out, 1)
        out = self.fuse(out)
        mask = my_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)
        return x * (1 - mask) + out * mask


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        inc = 3
        self.layer1 = nn.Sequential(
            spectral_norm(nn.Conv2d(inc, 64, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer3 = nn.Sequential(
            spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer4 = nn.Sequential(
            spectral_norm(nn.Conv2d(256, 512, 4, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer5 = nn.Conv2d(512, 1, 4, stride=1, padding=1)

        self.layer1.apply(init_weights)
        self.layer2.apply(init_weights)
        self.layer3.apply(init_weights)
        self.layer4.apply(init_weights)
        self.layer5.apply(init_weights)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feat = self.layer5(x)
        return feat


class PartialConvNet(nn.Module):
    # 256 x 256 image input, 256 = 2^8
    def __init__(self, input_size=256, layers=7):
        if 2 ** (layers + 1) != input_size:
            raise AssertionError

        super().__init__()
        self.freeze_enc_bn = False
        self.layers = layers

        # ======================= ENCODING LAYERS =======================
        # 3x256x256 --> 64x128x128
        self.encoder_1 = PartialConvLayer(3, 64, bn=False, sample="down-7")

        # 64x128x128 --> 128x64x64
        self.encoder_2 = PartialConvLayer(64, 128, sample="down-5")

        # 128x64x64 --> 256x32x32
        self.encoder_3 = PartialConvLayer(128, 256, sample="down-3")

        # 256x32x32 --> 512x16x16
        self.encoder_4 = PartialConvLayer(256, 512, sample="down-3")

        # 512x16x16 --> 512x8x8 --> 512x4x4 --> 512x2x2
        for i in range(5, layers + 1):
            name = "encoder_{:d}".format(i)
            setattr(self, name, PartialConvLayer(512, 512, sample="down-3"))

        # ======================= DECODING LAYERS =======================
        # dec_7: UP(512x2x2) + 512x4x4(enc_6 output) = 1024x4x4 --> 512x4x4
        # dec_6: UP(512x4x4) + 512x8x8(enc_5 output) = 1024x8x8 --> 512x8x8
        # dec_5: UP(512x8x8) + 512x16x16(enc_4 output) = 1024x16x16 --> 512x16x16
        for i in range(5, layers + 1):
            name = "decoder_{:d}".format(i)
            setattr(self, name, PartialConvLayer(512 + 512, 512, activation="leaky_relu"))

        # UP(512x16x16) + 256x32x32(enc_3 output) = 768x32x32 --> 256x32x32
        self.decoder_4 = PartialConvLayer(512 + 256, 256, activation="leaky_relu")

        # UP(256x32x32) + 128x64x64(enc_2 output) = 384x64x64 --> 128x64x64
        self.decoder_3 = PartialConvLayer(256 + 128, 128, activation="leaky_relu")

        # UP(128x64x64) + 64x128x128(enc_1 output) = 192x128x128 --> 64x128x128
        self.decoder_2 = PartialConvLayer(128 + 64, 64, activation="leaky_relu")

        # UP(64x128x128) + 3x256x256(original image) = 67x256x256 --> 3x256x256(final output)
        self.decoder_1 = PartialConvLayer(64 + 3, 3, bn=False, activation="", bias=True)

    def forward(self, input_x, mask):
        encoder_dict = {}
        mask_dict = {}

        key_prev = "h_0"
        encoder_dict[key_prev], mask_dict[key_prev] = input_x, mask

        for i in range(1, self.layers + 1):
            encoder_key = "encoder_{:d}".format(i)
            key = "h_{:d}".format(i)
            # Passes input and mask through encoding layer
            encoder_dict[key], mask_dict[key] = getattr(self, encoder_key)(encoder_dict[key_prev], mask_dict[key_prev])
            key_prev = key

        # Gets the final output data and mask from the encoding layers
        # 512 x 2 x 2
        out_key = "h_{:d}".format(self.layers)
        out_data, out_mask = encoder_dict[out_key], mask_dict[out_key]

        for i in range(self.layers, 0, -1):
            encoder_key = "h_{:d}".format(i - 1)
            decoder_key = "decoder_{:d}".format(i)

            # Upsample to 2 times scale, matching dimensions of previous encoding layer output
            out_data = F.interpolate(out_data, scale_factor=2)
            out_mask = F.interpolate(out_mask, scale_factor=2)

            # concatenate upsampled decoder output with encoder output of same H x W dimensions
            # s.t. final decoding layer input will contain the original image
            out_data = torch.cat([out_data, encoder_dict[encoder_key]], dim=1)
            # also concatenate the masks
            out_mask = torch.cat([out_mask, mask_dict[encoder_key]], dim=1)

            # feed through decoder layers
            out_data, out_mask = getattr(self, decoder_key)(out_data, out_mask)

        return out_data

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and "enc" in name:
                    # Sets batch normalization layers to evaluation mode
                    module.eval()


class PartialConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True, bias=False, sample="none-3", activation="relu"):
        super().__init__()
        self.bn = bn

        if sample == "down-7":
            # Kernel Size = 7, Stride = 2, Padding = 3
            self.input_conv = nn.Conv2d(in_channels, out_channels, 7, 2, 3, bias=bias)
            self.mask_conv = nn.Conv2d(in_channels, out_channels, 7, 2, 3, bias=False)

        elif sample == "down-5":
            self.input_conv = nn.Conv2d(in_channels, out_channels, 5, 2, 2, bias=bias)
            self.mask_conv = nn.Conv2d(in_channels, out_channels, 5, 2, 2, bias=False)

        elif sample == "down-3":
            self.input_conv = nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=bias)
            self.mask_conv = nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False)

        else:
            self.input_conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias)
            self.mask_conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)

        nn.init.constant_(self.mask_conv.weight, 1.0)

        # "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
        # negative slope of leaky_relu set to 0, same as relu
        # "fan_in" preserved variance from forward pass
        nn.init.kaiming_normal_(self.input_conv.weight, a=0, mode="fan_in")

        for param in self.mask_conv.parameters():
            param.requires_grad = False

        if bn:
            # Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
            # Applying BatchNorm2d layer after Conv will remove the channel mean
            self.batch_normalization = nn.BatchNorm2d(out_channels)

        if activation == "relu":
            # Used between all encoding layers
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            # Used between all decoding layers (Leaky RELU with alpha = 0.2)
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input_x, mask):
        # output = W^T dot (X .* M) + b
        output = self.input_conv(input_x * mask)

        # requires_grad = False
        with torch.no_grad():
            # mask = (1 dot M) + 0 = M
            output_mask = self.mask_conv(mask)

        if self.input_conv.bias is not None:
            # spreads existing bias values out along 2nd dimension (channels) and then expands to output size
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)

        # mask_sum is the sum of the binary mask at every partial convolution location
        mask_is_zero = (output_mask == 0)
        # temporarily sets zero values to one to ease output calculation
        mask_sum = output_mask.masked_fill_(mask_is_zero, 1.0)

        # output at each location as follows:
        # output = (W^T dot (X .* M) + b - b) / M_sum + b ; if M_sum > 0
        # output = 0 ; if M_sum == 0
        output = (output - output_bias) / mask_sum + output_bias
        output = output.masked_fill_(mask_is_zero, 0.0)

        # mask is updated at each location
        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(mask_is_zero, 0.0)

        if self.bn:
            output = self.batch_normalization(output)

        if hasattr(self, 'activation'):
            output = self.activation(output)

        return output, new_mask
