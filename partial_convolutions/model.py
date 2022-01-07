import torch
import torch.nn as nn
import torch.nn.functional as F

from partial_conv_layer import PartialConvLayer


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


if __name__ == '__main__':
    size = (1, 3, 256, 256)
    inp = torch.ones(size)
    input_mask = torch.ones(size)
    input_mask[:, :, 100:, :][:, :, :, 100:] = 0

    conv = PartialConvNet()
    l1 = nn.L1Loss()
    inp.requires_grad = True

    output = conv(inp, input_mask)
    loss = l1(output, torch.randn(1, 3, 256, 256))
    loss.backward()

    assert (torch.sum(inp.grad != inp.grad).item() == 0)
    assert (torch.sum(torch.isnan(conv.decoder_1.input_conv.weight.grad)).item() == 0)
    assert (torch.sum(torch.isnan(conv.decoder_1.input_conv.bias.grad)).item() == 0)
