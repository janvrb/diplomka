import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    """ 3D konvoluce """
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True)
    )

class ResBlock3D(nn.Module):
    """ Reziduální blok """
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
            nn.ReLU(inplace=True)
        )

        # Druhá část - Downsampling 2
        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )

        # Čtvrtá část - Upsampling 1 (přidává výstup z první části)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm3d(64),
            # nn.ReLU(inplace=True) # Podle článku zde není ReLU, ale až po přičtení shortcut (res connection)
        )

        # Pátá část - Upsampling 2 (přidává výstup z reziduálního bloku)
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm3d(32),
            # nn.ReLU(inplace=True)
        )

        # 1x1 konvoluce pro výstup z down1
        self.shortcut1 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(64),
            # nn.ReLU(inplace=True)
        )

        # 1x1 konvoluce pro x
        self.shortcut0 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(32),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        down1 = self.conv1(x)
        down2 = self.conv2(down1)

        # Padding pro srovnání rozlišení
        # print(f"self.shortcut0(x): {self.shortcut0(x).shape}")
        # pad_w = down2.shape[-1] - down1.shape[-1]
        # pad_h = down2.shape[-2] - down1.shape[-2]
        # if pad_w > 0 or pad_h > 0:
        #     down1 = F.pad(down1, (0, pad_w, 0, pad_h))

        up1_help2 = self.deconv1(down2)
        up1_help1 = self.shortcut1(down1)
        # output_padding1 = (up1_help1.shape[-2] - up1_help2.shape[-2], up1_help1.shape[-1] - up1_help2.shape[-1])
        # if output_padding1[0] >= 0 and output_padding1[0] >= 0:
        #     up1_help2 = F.pad(up1_help2, (0, output_padding1[1], 0, output_padding1[0]))
        # else:
        #     up1_help1 = F.pad(up1_help1, (0, -output_padding1[1], 0, -output_padding1[0]))

        up1_help2 = up1_help2[:, :, :up1_help1.size(2), :up1_help1.size(3), :up1_help1.size(4)]

        up1 = F.relu(up1_help2 + up1_help1, inplace=True)

        # print(f"self.deconv2(up1): {self.deconv2(up1).shape}")
        up2_help1 = self.deconv2(up1)
        up2_help2 = self.shortcut0(x)
        # output_padding2 = (up2_help1.shape[-2] - up2_help2.shape[-2], up2_help1.shape[-1] - up2_help2.shape[-1])
        # if output_padding2[0] >= 0 and output_padding2[0] >= 0:
        #     up2_help2 = F.pad(up2_help2, (0, output_padding2[1], 0, output_padding2[0]))
        # else:
        #     up2_help1 = F.pad(up2_help1, (0, -output_padding2[1], 0, -output_padding2[0]))

        up2_help1 = up2_help1[:, :, :up2_help2.size(2), :up2_help2.size(3), :up2_help2.size(4)]

        up2 = F.relu(up2_help1 + up2_help2, inplace=True)
        
        return up2  # Vracíme výstup pro další hourglass síť

    
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

    def forward(self, cost_volume, training=True):
        x = self.conv3d_1(cost_volume)
        x = self.conv3d_2(x)
        res_out = self.conv3d_3_4(x)

        # První Hourglass síť
        hg1_out = self.hourglass1(res_out)

        # Druhá Hourglass síť (přidává výstup z předchozí)
        hg2_out = self.hourglass2(hg1_out)

        # Třetí Hourglass síť (přidává výstup z předchozí)
        hg3_out = self.hourglass3(hg2_out)

        if training:
            out0 = self.final_conv(res_out)

            out1 = self.final_conv(hg1_out)

            out2 = self.final_conv(hg2_out)
        out3 = self.final_conv(hg3_out)

        if training:
            return [out0, out1, out2, out3]
        else:
            return [out3]
