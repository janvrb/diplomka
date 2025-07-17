import torch.nn as nn

class GwcNetLoss(nn.Module):
    """ Ztrátová funkce pro PSMNet (Smooth L1 Loss) """
    def __init__(self):
        super(GwcNetLoss, self).__init__()
        self.criterion = nn.SmoothL1Loss()

    def forward(self, disparity, ground_truth, train=True):

        mask = (ground_truth > 0) & (ground_truth < 192)  # Pouze platné disparity (KITTI má neznámé hodnoty jako 0)
        
        disp0 = disparity[0]
        disp1 = disparity[1]
        disp2 = disparity[2]
        disp3 = disparity[3]

        if train:
            loss0 = self.criterion(disp0[mask], ground_truth[mask])
            loss1 = self.criterion(disp1[mask], ground_truth[mask])
            loss2 = self.criterion(disp2[mask], ground_truth[mask])
            loss3 = self.criterion(disp3[mask], ground_truth[mask])
            return 0.5*loss0 + 0.5*loss1 + 0.7*loss2 + loss3
        else:
            return self.criterion(disp3[mask], ground_truth[mask])
