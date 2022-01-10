import torch.nn as nn

LAMBDAS = {"valid": 1.0, "hole": 6.0}


class CalculateLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, mask, output, ground_truth):
        loss_dict = dict()

        loss_dict["hole"] = self.l1((1 - mask) * output, (1 - mask) * ground_truth) * LAMBDAS["hole"]
        loss_dict["valid"] = self.l1(mask * output, mask * ground_truth) * LAMBDAS["valid"]

        return loss_dict
