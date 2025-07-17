import torch.nn as nn

class ACVNetLoss(nn.Module):
    """ Ztrátová funkce (Smooth L1 Loss) """
    def __init__(self, freeze=False, att_weights=False):
        super(ACVNetLoss, self).__init__()
        self.criterion = nn.SmoothL1Loss()

        self.freeze = freeze
        self.att_weights = att_weights

    def forward(self, disparity, ground_truth):

        mask = (ground_truth > 0) & (ground_truth < 192)  # Pouze platné disparity (KITTI má neznámé hodnoty jako 0)
        if not self.att_weights:
            disp0 = disparity[0]
            disp1 = disparity[1]
            disp2 = disparity[2]
            # print(f"mask shape: {mask.shape}, pred_disp shape: {disp0.shape}, gt_disp shape: {ground_truth.shape}")
            loss0 = self.criterion(disp0[mask], ground_truth[mask])
            loss1 = self.criterion(disp1[mask], ground_truth[mask])
            loss2 = self.criterion(disp2[mask], ground_truth[mask])

            if self.freeze:
                return 0.5*loss0 + 0.7*loss1 + loss2
            
            disp3 = disparity[3]
            loss3 = self.criterion(disp3[mask], ground_truth[mask])
            return 0.5*loss0 + 0.5*loss1 + 0.7*loss2 + loss3
        
        disp0 = disparity[0]
        loss0 = self.criterion(disp0[mask], ground_truth[mask])
        return loss0
        
