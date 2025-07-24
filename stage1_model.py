import torch
import random
import numpy as np
import torch.nn as nn
from thop import profile    
import torch.nn.functional as F
from resnet import resnet18,resnet34,resnet50


def SEED(Seed):
    seed = Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x


class adjust(nn.Module):
    def __init__(self, in_c1,in_c2,in_c3,in_c4, out_c):
        super().__init__()
        self.conv1 = CBR(in_c1, 64, kernel_size=1, padding=0, act=True)
        self.conv2 = CBR(in_c2, 64, kernel_size=1, padding=0, act=True)
        self.conv3 = CBR(in_c3, 64, kernel_size=1, padding=0, act=True)
        self.conv4 = CBR(in_c4, 64, kernel_size=1, padding=0, act=True)
        self.conv_fuse=nn.Conv2d(4*64, out_c, kernel_size=1, padding=0, bias=False)

    def forward(self, x11,x12,x13,x14,x21,x22,x23,x24):
        c1 = torch.cat([x11, x21], dim=1) ## [B,64*2,256,256]
        c2 = torch.cat([x12, x22], dim=1) ## [B,128*2,256,256]
        c3 = torch.cat([x13, x23], dim=1) ## [B,256*2,256,256]
        c4 = torch.cat([x14, x24], dim=1) ## [B,512*2,256,256]
        
        x1 = self.conv1(c1)  ## [B,64*2,256,256]
        x2 = self.conv2(c2)  ## [B,128*2,256,256]
        x3 = self.conv3(c3)  ## [B,256*2,256,256]
        x4 = self.conv4(c4)  ## [B,512*2,256,256]

        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.conv_fuse(x)
        return x


class CDStage1(nn.Module):
    def __init__(self):
        super().__init__()
        """ Backbone: ResNet18,34,50,Mobv2 """
        backbone = resnet18()
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # [batch_size, 64, h/2, w/2]
        self.layer1 = nn.Sequential(backbone.maxpool,backbone.layer1)  # [batch_size, 128, h/4, w/4]
        self.layer2 = backbone.layer2  # [batch_size, 256, h/8, w/8]
        self.layer3 = backbone.layer3  # [batch_size, 512, h/16, w/16]

        self.up_2x2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_4x4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.up_8x8 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        self.up_16x16 = nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True)

        self.head=adjust(128,256,512,1024,2)

    def forward(self, t1,t2):
        """ Backbone: ResNet50 """

        x11 = self.layer0(t1)   ## [-1, 64, h/2, w/2]
        x12 = self.layer1(x11)  ## [-1, 128, h/4, w/4]
        x13 = self.layer2(x12)  ## [-1, 256, h/8, w/8]
        x14 = self.layer3(x13)  ## [-1, 512, h/16, w/16]
        
        x21 = self.layer0(t2)   ## [-1, 64, h/2, w/2]
        x22 = self.layer1(x21)  ## [-1, 128, h/4, w/4]
        x23 = self.layer2(x22)  ## [-1, 256, h/8, w/8]
        x24 = self.layer3(x23)  ## [-1, 512, h/16, w/16]

        u11 = self.up_2x2(x11)  ## [B,64,256,256]
        u12 = self.up_4x4(x12)  ## [B,128,256,256]
        u13 = self.up_8x8(x13)  ## [B,256,256,256]
        u14 = self.up_16x16(x14)## [B,512,256,256]

        u21 = self.up_2x2(x21)  ## [B,64,256,256]
        u22 = self.up_4x4(x22)  ## [B,128,256,256]
        u23 = self.up_8x8(x23)  ## [B,256,256,256]
        u24 = self.up_16x16(x24)## [B,512,256,256]
        
        pred=self.head(u11,u12,u13,u14,u21,u22,u23,u24)

        return pred

if __name__ == '__main__':
    SEED(2025)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = DSP().to(device)

    t1 = torch.randn(1, 3, 256, 256).to(device)
    t2 = torch.randn(1, 3, 256, 256).to(device)

    out = model(t1,t2).to(device)

    flops, params = profile(model, inputs=(t1,t2))
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs", end=" | ")
    print(f"Params: {params / 1e6:.2f} M")

    print(out)
