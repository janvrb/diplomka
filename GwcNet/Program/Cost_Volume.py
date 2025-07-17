import torch
import torch.nn as nn

class CostVolume(nn.Module):

    def __init__(self, max_disparity, method='gwc', num_groups=40):
        super(CostVolume, self).__init__()
        self.max_disparity = max_disparity // 4  # Kvůli downsamplingu
        self.method = method
        self.num_groups = num_groups  # Počet skupin pro GwcNet

    def forward(self, left_gwc, right_gwc, left_unary, right_unary):
        if self.method == 'concat':
            B, C, H, W = left_unary.shape
            cost_volume = torch.zeros(B, C * 2, self.max_disparity, H, W, device=left_unary.device)
        else:
            B, C, H, W = left_gwc.shape
            cost_volume = torch.zeros(B, self.num_groups, self.max_disparity, H, W, device=left_gwc.device)
        
        for d in range(self.max_disparity):
            if self.method == 'concat':
                if d > 0:
                    shifted_right = torch.zeros_like(right_unary, device=right_unary.device)
                    shifted_right[:, :, :, :-d] = right_unary[:, :, :, d:]  # Posun doleva
                else:
                    shifted_right = right_unary
                # Concatenation-based (PSMNet)
                cost_volume[:, :, d, :, :] = torch.cat((left_unary, shifted_right), dim=1)
            else:
                # Group-wise correlation (GwcNet)
                if d > 0:
                    B2, C2, H2, W2 = left_gwc[:, :, :, d:].shape
                    cost_volume[:, :, d, :, d:] = (left_gwc[:, :, :, d:] * right_gwc[:, :, :, :-d]).view([B2, self.num_groups, C2 // self.num_groups, H2, W2]).mean(dim=2)
                else:
                    B2, C2, H2, W2 = left_gwc[:, :, :, d:].shape
                    cost_volume[:, :, d, :, :] = (left_gwc * right_gwc).view([B2, self.num_groups, C2 // self.num_groups, H2, W2]).mean(dim=2)
        
        return cost_volume.contiguous()
