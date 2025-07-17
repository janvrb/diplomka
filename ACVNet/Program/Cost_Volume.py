import torch
import torch.nn as nn
from Program.Cost_Aggregation import Hourglass3D, conv3d


class CostVolume(nn.Module):

    def __init__(self, max_disparity, method='gwc', num_groups=40):
        super(CostVolume, self).__init__()
        self.max_disparity = max_disparity // 4  # Kvůli downsamplingu
        self.method = method
        self.num_groups = num_groups  # Počet skupin pro GwcNet

        self.cost_volume_patch = nn.Conv3d(40, 40, kernel_size=(1,3,3), stride=1, dilation=1, groups=40, padding=(0,1,1), bias=False)
        self.cost_volume_patch_l1 = nn.Conv3d(8, 8, kernel_size=(1,3,3), stride=1, dilation=1, groups=8, padding=(0,1,1), bias=False)
        self.cost_volume_patch_l2 = nn.Conv3d(16, 16, kernel_size=(1,3,3), stride=1, dilation=2, groups=16, padding=(0,2,2), bias=False)
        self.cost_volume_patch_l3 = nn.Conv3d(16, 16, kernel_size=(1,3,3), stride=1, dilation=3, groups=16, padding=(0,3,3), bias=False)

        self.conv1 = nn.Sequential(conv3d(40, 32, kernel_size=3, stride=1, padding=1),
                                   nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm3d(32))
        
        self.hourglass = Hourglass3D()

        self.conv_final = nn.Sequential(conv3d(32, 32, kernel_size=3, stride=1, padding=1),
                                        nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=False))

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
                # Attention weights (ACVNet)
                if d > 0:
                    B2, C2, H2, W2 = left_gwc[:, :, :, d:].shape
                    cost_volume[:, :, d, :, d:] = (left_gwc[:, :, :, d:] * right_gwc[:, :, :, :-d]).view([B2, self.num_groups, C2 // self.num_groups, H2, W2]).mean(dim=2)
                else:
                    B2, C2, H2, W2 = left_gwc[:, :, :, d:].shape
                    cost_volume[:, :, d, :, :] = (left_gwc * right_gwc).view([B2, self.num_groups, C2 // self.num_groups, H2, W2]).mean(dim=2)

        if self.method == 'concat':
            return cost_volume.contiguous()
        else:
            # Úprava Gwc Volume podle ACVNet
            cost_volume = self.cost_volume_patch(cost_volume.contiguous())
            cost_volume_patch_l1 = self.cost_volume_patch_l1(cost_volume[:, :8])
            cost_volume_patch_l2 = self.cost_volume_patch_l2(cost_volume[:, 8:24])
            cost_volume_patch_l3 = self.cost_volume_patch_l3(cost_volume[:, 24:40])
            patch_cost_volume = torch.cat((cost_volume_patch_l1, cost_volume_patch_l2, cost_volume_patch_l3), dim=1)

            cost_volume_attention = self.conv1(patch_cost_volume)
            cost_volume_attention = self.hourglass(cost_volume_attention)
            cost_volume_attention_weights = self.conv_final(cost_volume_attention)
            return cost_volume_attention_weights
