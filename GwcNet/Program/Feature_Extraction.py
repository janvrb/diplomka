import torch
import torch.nn as nn
import torch.nn.functional as F

def conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation):
    """ Konvoluční vrstva """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                  stride=stride, padding=dilation if dilation > 1 else padding, 
                  dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channels)
    )

class ResBlock(nn.Module):
    """ Reziduální blok """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, downsample, padding, dilation):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Sequential(
            conv2d(in_channels, out_channels, 3, stride, padding, dilation),
            nn.ReLU(inplace=True)
        )
        self.conv2 = conv2d(out_channels, out_channels, 3, 1, padding, dilation)

        self.downsample = downsample  # Použije se, pokud se mění velikost vstupu
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)  # Downsampling vstupu

        out += identity  # Reziduální spojení (skip connection)
        # return F.relu(out)
        return out

class FeatureExtraction(nn.Module):
    def __init__(self):
        super(FeatureExtraction, self).__init__()
        self.in_channels = 32  # Počet kanálů na začátku

        # První konvoluce
        self.conv_start = nn.Sequential(
            conv2d(3, 32, 3, 2, 1, 1),
            nn.ReLU(inplace=True),
            conv2d(32, 32, 3, 1, 1, 1),
            nn.ReLU(inplace=True),
            conv2d(32, 32, 3, 1, 1, 1),
            nn.ReLU(inplace=True)
        )

        # Reziduální bloky
        self.layer1 = self.make_res_layer(ResBlock, 32, 3, 1, 1, 1)
        self.layer2 = self.make_res_layer(ResBlock, 64, 16, 2, 1, 1)  # Downsampling (stride=2)
        self.layer3 = self.make_res_layer(ResBlock, 128, 3, 1, 1, 1)
        self.layer4 = self.make_res_layer(ResBlock, 128, 3, 1, 1, 2)  # Dilated konvoluce


        # Finální konvoluce
        self.conv_final = nn.Sequential(
            conv2d(320, 128, 3, 1, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 12, kernel_size=1, padding=0, stride=1, bias=False)
        )

    def make_res_layer(self, block, out_channels, num_blocks, stride, padding, dilation):
        """ Vytvoření vrstvy s více reziduálními bloky """
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, padding, dilation))  # První blok s downsamplingem
        self.in_channels = out_channels * block.expansion  # Aktualizace počtu kanálů

        for _ in range(1, num_blocks): 
            layers.append(block(self.in_channels, out_channels, 1, None, padding, dilation))

        return nn.Sequential(*layers) 

    def forward(self, x):
        x = self.conv_start(x)  # První konvoluce

        x = self.layer1(x)  # První sada reziduálních bloků
        x2 = self.layer2(x)  # Druhá sada (se stride=2)
        x3 = self.layer3(x2)  # Třetí sada
        x4 = self.layer4(x3)  # Čtvrtá sada
        gwc_final = torch.cat([x2, x3, x4], dim=1) # Spojení všech feature map

        gwc_features = gwc_final  # Featury pro Gwc korelaci
        unary_features = self.conv_final(gwc_final)  # Unární featury
        
        return gwc_features, unary_features
    

