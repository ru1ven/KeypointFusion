import torch

class SmoothL1Loss(torch.nn.Module):
    def __init__(self, size_average=True):
        super(SmoothL1Loss, self).__init__()
        self.size_average = size_average

    def forward(self, x, y):
        total_loss = 0
        assert(x.shape == y.shape)
        z = (x - y).float()
        mse_mask = (torch.abs(z) < 0.01).float()
        l1_mask = (torch.abs(z) >= 0.01).float()
        mse = mse_mask * z
        l1 = l1_mask * z
        total_loss += torch.mean(self._calculate_MSE(mse)*mse_mask, dim=-1)
        total_loss += torch.mean(self._calculate_L1(l1)*l1_mask, dim=-1)
        if self.size_average:
            return total_loss.mean()
        else:
            return total_loss.sum()

    def _calculate_MSE(self, z):
        return 0.5 *(torch.pow(z, 2))

    def _calculate_L1(self, z):
        return 0.01 * (torch.abs(z) - 0.005)