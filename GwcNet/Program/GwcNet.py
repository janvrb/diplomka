import torch
import torch.nn as nn
import torch.nn.functional as F
from Program.Feature_Extraction import FeatureExtraction
from Program.Cost_Volume import CostVolume
from Program.Cost_Aggregation import CostAggregation
from Program.Disparity_Regression import DisparityRegression

class GwcNet(nn.Module):
    def __init__(self, max_disparity=192):
        super(GwcNet, self).__init__()
        self.max_disparity = max_disparity

        self.feature_extraction = FeatureExtraction()

        self.cost_volume_concat = CostVolume(max_disparity, method='concat')
        self.cost_volume_gwc = CostVolume(max_disparity, method='gwc')

        self.cost_aggregation = CostAggregation()

        self.disparity_regression = DisparityRegression(max_disparity)

    def forward(self, left_img, right_img, methods='concat', training=True):
        # Extrakce featur
        left_gwc, left_unary = self.feature_extraction(left_img)
        right_gwc, right_unary = self.feature_extraction(right_img)

        # Výpočet cost volume
        volume_gwc = self.cost_volume_gwc(left_gwc, right_gwc, left_unary, right_unary)
        if methods == 'concat':
            volume_concat = self.cost_volume_concat(left_gwc, right_gwc, left_unary, right_unary)
            cost_volume = torch.cat((volume_gwc, volume_concat), 1)
        else:
            cost_volume = volume_gwc

        # Regularizace cost volume
        regularized_costs = self.cost_aggregation(cost_volume, training)

        upsampled_costs = [
            F.interpolate(cost, 
                          size=(self.max_disparity, left_img.shape[-2], left_img.shape[-1]), 
                          mode='trilinear', 
                          align_corners=False)
            for cost in regularized_costs
        ]

        softmax_cost = [F.softmax(torch.squeeze(cost, 1), dim=1) for cost in upsampled_costs]

        # Disparity Regression pro každý výstup z Hourglass sítí
        disparity = [self.disparity_regression(cost) for cost in softmax_cost]

        disparity_final = [torch.squeeze(cost, 1) for cost in disparity]

        return disparity_final

