import torch
import torch.nn as nn

class CostVolume(nn.Module):
    """ Konstrukce cost volume posunem pravého obrazu a zřetězením feature map """
    def __init__(self, max_disparity):
        super(CostVolume, self).__init__()
        self.max_disparity = max_disparity // 4  # Kvůli downsamplingu ve FeatureExtraction

    def forward(self, left_features, right_features):
        B, C, H, W = left_features.shape
        cost_volume = torch.zeros(B, C * 2, self.max_disparity, H, W).to(left_features.device)

        for d in range(self.max_disparity):
            if d > 0:
                shifted_right = torch.zeros_like(right_features).to(right_features.device)
                shifted_right[:, :, :, :-d] = right_features[:, :, :, d:]  
                cost_volume[:, :, d, :, :] = torch.cat((left_features, shifted_right), dim=1)
            else:
                cost_volume[:, :, d, :, :] = torch.cat((left_features, right_features), dim=1)

        return cost_volume
