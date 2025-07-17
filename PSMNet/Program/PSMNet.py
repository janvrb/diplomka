import torch
import torch.nn as nn
import torch.nn.functional as F
from Program.Feature_Extraction import FeatureExtraction
from Program.Cost_Volume import CostVolume
from Program.Cost_Aggregation import CostAggregation
from Program.Disparity_Regression import DisparityRegression

class PSMNet(nn.Module):
    def __init__(self, max_disparity=192):
        super(PSMNet, self).__init__()
        self.max_disparity = max_disparity

        self.feature_extraction = FeatureExtraction()

        self.cost_volume = CostVolume(max_disparity)

        self.cost_aggregation = CostAggregation()

        self.disparity_regression = DisparityRegression(max_disparity)

    def forward(self, left_img, right_img):
        # Extrakce featur
        left_features = self.feature_extraction(left_img)
        right_features = self.feature_extraction(right_img)

        # Konstrukce cost volume
        cost_volume = self.cost_volume(left_features, right_features)

        # Regularizace cost volume
        regularized_costs = self.cost_aggregation(cost_volume)

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

        return disparity

