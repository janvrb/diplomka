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
            conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        )

        # Druhá část - Downsampling 2
        self.conv2 = nn.Sequential(
            conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            conv3d(128, 128, kernel_size=3, stride=1, padding=1)
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

        self.att_block = AttentionBlock(channels_3d=128, num_heads=16, block_size=(4, 4, 4))


    def forward(self, x):
        down1 = self.conv1(x)
        down2 = self.conv2(down1)
        down2 = self.att_block(down2)

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

        # Dvě úvodní 3D konvoluce (3x3x3, 32)
        self.conv3d_1 = conv3d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv3d_2 = conv3d(32, 32, kernel_size=3, stride=1, padding=1)

        # Reziduální blok (dvě konvoluce 3x3x3, 32)
        # self.conv3d_3_4 = ResBlock3D(32)
        self.conv3d_3_4 = nn.Sequential(
            conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32)
        )

        # Dvě hourglass sítě
        self.hourglass1 = Hourglass3D()
        self.hourglass2 = Hourglass3D()

        # Výstupní konvoluce (snížení na 1 kanál)
        self.final_conv1 = nn.Sequential(
            conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=False)
        )

        self.final_conv2 = nn.Sequential(
            conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=False)
        )

        self.final_conv3 = nn.Sequential(
            conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, cost_volume, training=True):
        x = self.conv3d_1(cost_volume)
        x = self.conv3d_2(x)
        res_out = self.conv3d_3_4(x) + x

        # První Hourglass síť
        hg1_out = self.hourglass1(res_out)
        # Druhá Hourglass síť (přidává výstup z předchozí)
        hg2_out = self.hourglass2(hg1_out)

        if training:
            out0 = self.final_conv1(res_out)

            out1 = self.final_conv2(hg1_out)

        out2 = self.final_conv3(hg2_out)

        if training:
            return [out0, out1, out2]
        else:
            return [out2]

class AttentionBlock(nn.Module):
    def __init__(self, channels_3d, num_heads=8, block_size=(4, 4, 4)):

        super(AttentionBlock, self).__init__()
        self.block_size = block_size
        self.channels_3d = channels_3d
        self.num_heads = num_heads
        head_dim = channels_3d // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(channels_3d, channels_3d * 3, bias=True)
        self.final_conv = nn.Conv3d(channels_3d, channels_3d, kernel_size=1)

    def forward(self, x):
        B, C, D, H0, W0 = x.shape
        pad_D = (self.block_size[0] - D % self.block_size[0]) % self.block_size[0]
        pad_H = (self.block_size[1] - H0 % self.block_size[1]) % self.block_size[1]
        pad_W = (self.block_size[2] - W0 % self.block_size[2]) % self.block_size[2]

        # Padování pro rozdělení do bloků
        x = F.pad(x, (0, pad_W, 0, pad_H, 0, pad_D))

        B, C, D, H, W = x.shape
        d, h, w = D // self.block_size[0], H // self.block_size[1], W // self.block_size[2]

        # Přeuspořádání do bloků
        x = x.view(B, C, d, self.block_size[0], h, self.block_size[1], w, self.block_size[2])
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1)  # [B, d, h, w, bd, bh, bw, C]
        x = x.reshape(B, d * h * w, self.block_size[0] * self.block_size[1] * self.block_size[2], C)

        # QKV projekce
        qkv = self.qkv(x).reshape(B, d * h * w, self.block_size[0] * self.block_size[1] * self.block_size[2], 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(3, 0, 1, 4, 2, 5)  # [3, B, d*h*w, heads, num_points_in_block, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Výpočet attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)

        # Aplikace attention
        x = (attn @ v)
        x = x.permute(0, 1, 3, 2, 4).reshape(B, d, h, w, self.block_size[0], self.block_size[1], self.block_size[2], C)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, C, D, H, W)

        # Odstraníme padding
        if pad_D > 0:
            x = x[:, :, :-pad_D, :, :]
        if pad_H > 0:
            x = x[:, :, :, :-pad_H, :]
        if pad_W > 0:
            x = x[:, :, :, :, :-pad_W]

        # Finální 1×1×1 konvoluce
        x = self.final_conv(x)

        return x