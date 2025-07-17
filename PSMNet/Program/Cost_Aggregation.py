import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    """ 3D konvoluce + BatchNorm """
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True)
    )

class ResBlock3D(nn.Module):
    """ Reziduální blok: Conv3D → BatchNorm → ReLU → Conv3D → BatchNorm + skip connection """
    def __init__(self, channels):
        super(ResBlock3D, self).__init__()

        self.conv1 = conv3d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn(self.conv2(out))
        out += identity  # Skip connection
        return F.relu(out)

class Hourglass3D(nn.Module):
    """ Jedna Hourglass síť pro 3D Cost Aggregation """
    def __init__(self):
        super(Hourglass3D, self).__init__()

        # První část - Downsampling 1
        self.conv1 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(64),
            # nn.ReLU(inplace=True)
        )

        # Druhá část - Downsampling 2
        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )

        # Čtvrtá část - Upsampling 1 (přidává výstup z první části)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm3d(64),
            # nn.ReLU(inplace=True)
        )

        # Pátá část - Upsampling 2 (přidává výstup z reziduálního bloku)
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm3d(32),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x, con1, con2, not_first=False):
        if not_first:
            down1 = F.relu(self.conv1(x) + con1, inplace=True)
        else:
            down1 = F.relu(self.conv1(x), inplace=True)
        down2 = self.conv2(down1)

        if not_first:
            up1 = self.deconv1(down2)
            output_padding1 = (con2.shape[-2] - up1.shape[-2], con2.shape[-1] - up1.shape[-1])
            up1 = F.pad(up1, (0, output_padding1[1], 0, output_padding1[0])) + con2  
            up1 = F.relu(up1 + con2, inplace=True)  # Přidáváme výstup z první části
        else:
            up1 = self.deconv1(down2)
            output_padding2 = (down1.shape[-2] - up1.shape[-2], down1.shape[-1] - up1.shape[-1])
            up1 = F.pad(up1, (0, output_padding2[1], 0, output_padding2[0])) + down1 
            up1 = F.relu(up1 + down1 , inplace=True)
        up2 = self.deconv2(up1)  # Přidáváme výstup z reziduálního bloku
        
        return up2, down1, up1  # Vracíme výstup + mezivýstup pro další hourglass síť

    
class CostAggregation(nn.Module):
    def __init__(self):
        super(CostAggregation, self).__init__()

        # Dvě 3D konvoluce
        self.conv3d_1 = conv3d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv3d_2 = conv3d(32, 32, kernel_size=3, stride=1, padding=1)

        # Reziduální blok
        # self.res_block = ResBlock3D(32)
        self.conv3d_3_4 = nn.Sequential(
            conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32)
        )

        # Tři hourglass sítě
        self.hourglass1 = Hourglass3D()
        self.hourglass2 = Hourglass3D()
        self.hourglass3 = Hourglass3D()

        # Výstupní konvoluce (snížení na 1 kanál)
        self.final_conv = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, cost_volume):
        x = self.conv3d_1(cost_volume)
        x = self.conv3d_2(x)
        res_out = self.conv3d_3_4(x)

        # První Hourglass síť
        hg1_out, hg1_down1, hg1_up1 = self.hourglass1(res_out, res_out, res_out, False)

        # Přidání padding kvůli rozdílným šířkám/výškám/disparitám kvůli konvolucím a dekonvolucím lichých hodnot v Hourglass
        # pad = (res_out.shape[-1] - hg1_out.shape[-1], res_out.shape[-2] - hg1_out.shape[-2], res_out.shape[-3] - hg1_out.shape[-3])
        # if pad[0] > 0:
        #     hg1_out = F.pad(hg1_out, (0, pad[0]))  # Přidáme padding na pravou stranu
        # elif pad[0] < 0:
        #     res_out = F.pad(res_out, (0, -pad[0]))  # Přidáme padding na res_out
        # if pad[1] > 0:
        #     hg1_out = F.pad(hg1_out, (0, 0, 0, pad[1]))  # Přidáme padding nahoru
        # elif pad[1] < 0:
        #     res_out = F.pad(res_out, (0, 0, 0, -pad[1]))  # Přidáme padding na res_out
        # if pad[2] > 0:
        #     hg1_out = F.pad(hg1_out, (0, 0, 0, 0, 0, pad[2]))  # Přidáme padding do hloubky
        # elif pad[2] < 0:
        #     res_out = F.pad(res_out, (0, 0, 0, 0, 0, -pad[2]))  # Přidáme padding na res_out

        hg1_out = hg1_out[:, :, :res_out.size(2), :res_out.size(3), :res_out.size(4)]

        hg1_out = hg1_out + res_out

        # Druhá Hourglass síť (přidává výstup z předchozí)
        hg2_out, _, hg1_up2 = self.hourglass2(hg1_out, hg1_up1, hg1_down1, True)
        hg2_out = hg2_out[:, :, :res_out.size(2), :res_out.size(3), :res_out.size(4)]
        hg2_out = hg2_out + res_out

        # Třetí Hourglass síť (přidává výstup z předchozí)
        hg3_out, _, _ = self.hourglass3(hg2_out, hg1_up2, hg1_down1, True)
        hg3_out = hg3_out[:, :, :res_out.size(2), :res_out.size(3), :res_out.size(4)]
        hg3_out = hg3_out + res_out

        out1 = self.final_conv(hg1_out)

        out2 = self.final_conv(hg2_out)
        out2 = out2 + out1

        out3 = self.final_conv(hg3_out)
        out3 = out3 + out2

        return [out1, out2, out3]
