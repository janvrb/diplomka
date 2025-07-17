import torch
import torch.nn as nn
import torch.nn.functional as F

class DisparityRegression(nn.Module):
    """ Disparity Regression pomocí Soft Argmax """
    def __init__(self, max_disparity):
        super(DisparityRegression, self).__init__()
        self.max_disparity = max_disparity
        self.register_buffer("disparity_values", torch.arange(0, max_disparity, dtype=torch.float32).view(1, max_disparity, 1, 1))

    def forward(self, prob_volume):
        """ Výpočet disparity jako vážený průměr přes pravděpodobnosti """
        disparity_map = torch.sum(prob_volume * self.disparity_values, dim=1, keepdim=True)  
        return disparity_map  

