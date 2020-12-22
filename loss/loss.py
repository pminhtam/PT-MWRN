import torch.nn as nn


class BasicLoss(nn.Module):
    """
    Basic loss function.
    """
    def __init__(self):
        super(BasicLoss, self).__init__()
        self.l2_loss = nn.MSELoss()

    def forward(self, pred, ground_truth):
        return self.l2_loss(pred, ground_truth)
