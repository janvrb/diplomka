import torch
import torch.nn as nn
import torch.nn.functional as F
from Program.Feature_Extraction import FeatureExtraction
from Program.Cost_Volume import CostVolume
from Program.Cost_Aggregation import CostAggregation
from Program.Disparity_Regression import DisparityRegression

class ACVNet(nn.Module):
    def __init__(self, max_disparity=192, training=True, freeze_att_weights=False, att_weights_only=False):
        super(ACVNet, self).__init__()
        self.max_disparity = max_disparity

        self.feature_extraction = FeatureExtraction()

        self.cost_volume_concat = CostVolume(max_disparity, method='concat')
        self.cost_volume_gwc = CostVolume(max_disparity, method='gwc')

        self.cost_aggregation = CostAggregation()

        self.disparity_regression = DisparityRegression(max_disparity)

        self.training = training
        self.freeze_att_weights = freeze_att_weights
        self.att_weights_only = att_weights_only

    def forward(self, left_img, right_img):
        if self.freeze_att_weights: # Zamrzlé výpočty gradientů pro attention weights
            with torch.no_grad():
                # Extrakce featur
                left_gwc, left_unary = self.feature_extraction(left_img)
                right_gwc, right_unary = self.feature_extraction(right_img)
                # Výpočet cost volume pomocí Attention concatenation volume
                attention_weights = self.cost_volume_gwc(left_gwc, right_gwc, left_unary, right_unary)
        else:
            # Extrakce featur
            left_gwc, left_unary = self.feature_extraction(left_img)
            right_gwc, right_unary = self.feature_extraction(right_img)
            # Výpočet cost volume pomocí Attention concatenation volume
            attention_weights = self.cost_volume_gwc(left_gwc, right_gwc, left_unary, right_unary)

        if not self.att_weights_only:
            volume_concat = self.cost_volume_concat(left_gwc, right_gwc, left_unary, right_unary)
            attention_weights = F.interpolate(attention_weights, 
                          size=(volume_concat.shape[-3], volume_concat.shape[-2], volume_concat.shape[-1]), 
                          mode='trilinear', 
                          align_corners=False) # Interpolace att_weights na stejné rozměry jako concatenation volume
            cost_volume = F.softmax(attention_weights, dim=2) * volume_concat # Vytvoření Attention concatenation volume spojením upravené Gwc a concat
            # Regularizace cost volume
            regularized_costs = self.cost_aggregation(cost_volume, self.training)

        if self.training: # Por trénování sítě
            if not self.freeze_att_weights:
                # Disparity regression pro attention weights
                upsampled_attention_weights = F.interpolate(attention_weights, 
                                size=(self.max_disparity, left_img.shape[-2], left_img.shape[-1]), 
                                mode='trilinear', 
                                align_corners=False)
                softmax_attention_weights = F.softmax(torch.squeeze(upsampled_attention_weights, 1), dim=1)
                pred_attention = self.disparity_regression(softmax_attention_weights)
                pred_attention_final = torch.squeeze(pred_attention, 1)

            if not self.att_weights_only:
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

                if self.freeze_att_weights:
                    return [disparity_final[0], disparity_final[1], disparity_final[2]]
                return [pred_attention_final, disparity_final[0], disparity_final[1], disparity_final[2]]
            return [pred_attention_final]
        
        else: # Pro testování nebo validování sítě
            if self.att_weights_only:
                # Disparity regression pro attention weights
                upsampled_attention_weights = F.interpolate(attention_weights, 
                                size=(self.max_disparity, left_img.shape[-2], left_img.shape[-1]), 
                                mode='trilinear', 
                                align_corners=False)
                softmax_attention_weights = F.softmax(torch.squeeze(upsampled_attention_weights, 1), dim=1)
                pred_attention = self.disparity_regression(softmax_attention_weights)
                pred_attention_final = torch.squeeze(pred_attention, 1)
                return [pred_attention_final]

            upsampled_costs = F.interpolate(regularized_costs[0], 
                                size=(self.max_disparity, left_img.shape[-2], left_img.shape[-1]), 
                                mode='trilinear', 
                                align_corners=False)
 
            softmax_cost = F.softmax(torch.squeeze(upsampled_costs, 1), dim=1)
            # Disparity Regression pro každý výstup z Hourglass sítí
            disparity = self.disparity_regression(softmax_cost)
            disparity_final = torch.squeeze(disparity[0], 1)
            return [disparity_final]

