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
        # 同尺度拼接
        c1 = torch.cat([x11, x21], dim=1) ## [B,64*2,256,256]
        c2 = torch.cat([x12, x22], dim=1) ## [B,128*2,256,256]
        c3 = torch.cat([x13, x23], dim=1) ## [B,256*2,256,256]
        c4 = torch.cat([x14, x24], dim=1) ## [B,512*2,256,256]
        
        x1 = self.conv1(c1)  ## [B,64*2,256,256]
        x2 = self.conv2(c2)  ## [B,128*2,256,256]
        x3 = self.conv3(c3)  ## [B,256*2,256,256]
        x4 = self.conv4(c4)  ## [B,512*2,256,256]
        # 多尺度融合
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.conv_fuse(x)
        return x

class FeatureFusionModule(nn.Module):
    def __init__(self, in_d, out_d):
        super(FeatureFusionModule, self).__init__()
        self.conv = nn.Conv2d(in_d, out_d, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(out_d)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class NeighborFeatureAggregation(nn.Module):
    def __init__(self, in_d=[64, 128, 256, 512]):
        super(NeighborFeatureAggregation, self).__init__()
        self.in_d = in_d

        # s1 (对应 c2)
        self.conv_s1_c2 = nn.Sequential(
            nn.Conv2d(in_d[0], in_d[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_d[0]),
            nn.ReLU(inplace=True)
        )
        self.conv_s1_c3 = nn.Sequential(
            nn.Conv2d(in_d[1], in_d[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_d[0]),
            nn.ReLU(inplace=True)
        )
        self.aggregation_s1 = FeatureFusionModule(in_d[0] * 2, in_d[0])

        # s2 (对应 c3)
        self.conv_s2_c2 = nn.Sequential(
            nn.Conv2d(in_d[0], in_d[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_d[1]),
            nn.ReLU(inplace=True)
        )
        self.conv_s2_c3 = nn.Sequential(
            nn.Conv2d(in_d[1], in_d[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_d[1]),
            nn.ReLU(inplace=True)
        )
        self.conv_s2_c4 = nn.Sequential(
            nn.Conv2d(in_d[2], in_d[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_d[1]),
            nn.ReLU(inplace=True)
        )
        self.aggregation_s2 = FeatureFusionModule(in_d[1] * 3, in_d[1])

        # s3 (对应 c4)
        self.conv_s3_c3 = nn.Sequential(
            nn.Conv2d(in_d[1], in_d[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_d[2]),
            nn.ReLU(inplace=True)
        )
        self.conv_s3_c4 = nn.Sequential(
            nn.Conv2d(in_d[2], in_d[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_d[2]),
            nn.ReLU(inplace=True)
        )
        self.conv_s3_c5 = nn.Sequential(
            nn.Conv2d(in_d[3], in_d[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_d[2]),
            nn.ReLU(inplace=True)
        )
        self.aggregation_s3 = FeatureFusionModule(in_d[2] * 3, in_d[2])

        # s4 (对应 c5)
        self.conv_s4_c4 = nn.Sequential(
            nn.Conv2d(in_d[2], in_d[3], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_d[3]),
            nn.ReLU(inplace=True)
        )
        self.conv_s4_c5 = nn.Sequential(
            nn.Conv2d(in_d[3], in_d[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_d[3]),
            nn.ReLU(inplace=True)
        )
        self.aggregation_s4 = FeatureFusionModule(in_d[3] * 2, in_d[3])

    def forward(self, c2, c3, c4, c5):
        # s1
        c2_s1 = self.conv_s1_c2(c2)
        c3_s1 = self.conv_s1_c3(F.interpolate(c3, scale_factor=2, mode='bilinear', align_corners=False))
        s1 = self.aggregation_s1(torch.cat([c2_s1, c3_s1], dim=1))

        # s2
        c2_s2 = self.conv_s2_c2(c2)
        c3_s2 = self.conv_s2_c3(c3)
        c4_s2 = self.conv_s2_c4(F.interpolate(c4, scale_factor=2, mode='bilinear', align_corners=False))
        s2 = self.aggregation_s2(torch.cat([c2_s2, c3_s2, c4_s2], dim=1))

        # s3
        c3_s3 = self.conv_s3_c3(c3)
        c4_s3 = self.conv_s3_c4(c4)
        c5_s3 = self.conv_s3_c5(F.interpolate(c5, scale_factor=2, mode='bilinear', align_corners=False))
        s3 = self.aggregation_s3(torch.cat([c3_s3, c4_s3, c5_s3], dim=1))

        # s4
        c4_s4 = self.conv_s4_c4(c4)
        c5_s4 = self.conv_s4_c5(c5)
        s4 = self.aggregation_s4(torch.cat([c4_s4, c5_s4], dim=1))

        return s1, s2, s3, s4
    
class LightweightFusionModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(LightweightFusionModule, self).__init__()
        # 通道注意力模块（轻量级设计）
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        
        # 空间注意力模块（卷积方式）
        self.conv_spatial = nn.Conv2d(in_channels * 2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat1, feat2):
        B, C, H, W = feat1.size()

        # 拼接两个时相特征图
        combined = torch.cat([feat1, feat2], dim=1)  # [B, 2C, H, W]
        
        # ----- 通道注意力分支 -----
        avg_pool = F.adaptive_avg_pool2d(combined, 1).view(B, -1)
        channel_attn = self.fc(avg_pool)  # [B, C]
        channel_attn = self.sigmoid(channel_attn).view(B, C, 1, 1)
        
        # ----- 空间注意力分支 -----
        spatial_attn = self.sigmoid(self.conv_spatial(combined))  # [B, 1, H, W]
        
        # ----- 融合注意力 -----
        final_attn = channel_attn * spatial_attn  # [B, 1, H, W]

        # ----- 加权融合 -----
        fused = (1 - final_attn) * feat1 + final_attn * feat2
        
        return fused, final_attn
    
class EntropyGuidedFusion(nn.Module):
    """ 熵引导自适应融合模块（EGAFM） """
    def __init__(self, beta=5.0, epsilon=1e-6):
        super(EntropyGuidedFusion, self).__init__()
        self.beta = beta
        self.epsilon = epsilon

    def compute_entropy(self, feat):
        """ 计算局部通道熵 """
        prob = F.softmax(feat, dim=1)  
        entropy = -torch.sum(prob * torch.log(prob + self.epsilon), dim=1, keepdim=True)
        return entropy

    def forward(self, feat_A, feat_B):
        entropy_A = self.compute_entropy(feat_A)
        entropy_B = self.compute_entropy(feat_B)
        diff = entropy_A - entropy_B
        weight_B = torch.sigmoid(self.beta * diff)  
        fused = (1 - weight_B) * feat_A + weight_B * feat_B
        return fused, weight_B

class DualEntropyGuidedAttentionFusion(nn.Module):
    def __init__(self, c, h):
        super(DualEntropyGuidedAttentionFusion, self).__init__()
        self.egafm = EntropyGuidedFusion(beta=5.0)

    def forward(self, feat_A, feat_B):
        b,c,h,w = feat_A.shape
        fused, weight_B = self.egafm(feat_A, feat_B)
        return fused, weight_B

class LowLevelGatedFusion(nn.Module):
    def __init__(self, high_channels, low_channels):
        super(LowLevelGatedFusion, self).__init__()

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(low_channels * 2, low_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_channels),
            nn.ReLU(inplace=True)
        )

        self.high_map = nn.Sequential(
        nn.Conv2d(high_channels, low_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(low_channels),
        nn.ReLU(inplace=True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, high_feat, low_feat1,low_feat2):
        high_feat_up = F.interpolate(high_feat, size=low_feat1.shape[2:], mode='bilinear', align_corners=False)
        highfeat = self.high_map(high_feat_up)
        gate = self.sigmoid(highfeat) 
        low_feat = torch.cat([low_feat1,low_feat2],dim=1)
        lowfeat = self.fuse_conv(low_feat)
        low_feat_filtered = gate * lowfeat 

        fused = highfeat + low_feat_filtered
        return fused

class HighLevelSemanticFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HighLevelSemanticFusion, self).__init__()
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, feat_t1, feat_t2):
        # feat_t1, feat_t2: [B, in_channels, H, W]
        fused = torch.cat([feat_t1, feat_t2], dim=1)  # [B, 2*in_channels, H, W]
        fused = self.fuse_conv(fused)  # [B, out_channels, H, W]
        return fused

class GuidanceFusionBlock(nn.Module):
    """
    指导融合块：将当前尺度特征与上采样后的高层语义特征进行融合
    """
    def __init__(self, in_channels, guidance_channels, out_channels):
        super(GuidanceFusionBlock, self).__init__()
        # 先将高层语义特征通过1x1映射到与当前尺度特征相同通道数
        self.guidance_conv = nn.Sequential(
            nn.Conv2d(guidance_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        # 融合：拼接后通过3x3卷积降维
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, feat, guidance):
        # guidance 已经上采样到与 feat 相同的尺寸
        guidance_mapped = self.guidance_conv(guidance)
        fused = torch.cat([feat, guidance_mapped], dim=1)
        fused = self.fuse_conv(fused)
        return fused

class SupervisedAttentionModule(nn.Module):
    def __init__(self, mid_d):
        super(SupervisedAttentionModule, self).__init__()
        self.mid_d = mid_d
        # 注意力机制相关模块
        self.cls = nn.Conv2d(self.mid_d, 1, kernel_size=1)
        self.conv_context = nn.Sequential(
            nn.Conv2d(2, self.mid_d, kernel_size=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        mask = self.cls(x)
        mask_f = torch.sigmoid(mask)
        mask_b = 1 - mask_f
        context = torch.cat([mask_f, mask_b], dim=1)
        context = self.conv_context(context)
        x = x.mul(context)
        x_out = self.conv2(x)

        return x_out, mask

class Decoder(nn.Module):
    def __init__(self, mid_d=128,high_semantic_channels=256):
        super(Decoder, self).__init__()
        self.mid_d = mid_d

        self.conv_d2 = nn.Conv2d(64, mid_d, kernel_size=1)
        self.conv_d3 = nn.Conv2d(128, mid_d, kernel_size=1)
        self.conv_d4 = nn.Conv2d(256, mid_d, kernel_size=1)
        self.conv_d5 = nn.Conv2d(512, mid_d, kernel_size=1)
        

        self.sam_p5 = SupervisedAttentionModule(mid_d)
        self.sam_p4 = SupervisedAttentionModule(mid_d)
        self.sam_p3 = SupervisedAttentionModule(mid_d)

        self.conv_p4 = nn.Sequential(
            nn.Conv2d(mid_d, mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_p3 = nn.Sequential(
            nn.Conv2d(mid_d, mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(mid_d, mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_d),
            nn.ReLU(inplace=True)
        )
        self.cat_p2 = nn.Sequential(
            nn.Conv2d(mid_d*2, mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_d),
            nn.ReLU(inplace=True)
        )
        self.cls = nn.Conv2d(mid_d, 1, kernel_size=1)

        self.fusion_block_p5 = GuidanceFusionBlock(in_channels=mid_d, guidance_channels=high_semantic_channels, out_channels=mid_d)
        self.fusion_block_p4 = GuidanceFusionBlock(in_channels=mid_d, guidance_channels=high_semantic_channels, out_channels=mid_d)
        self.fusion_block_p3 = GuidanceFusionBlock(in_channels=mid_d, guidance_channels=high_semantic_channels, out_channels=mid_d)
        self.fusion_block_p2 = GuidanceFusionBlock(in_channels=mid_d, guidance_channels=high_semantic_channels, out_channels=mid_d)
        

    def forward(self, d2, d3, d4, d5,high_semantic,low_feature):
        d2 = self.conv_d2(d2)  # [B, mid_d, 128, 128]
        d3 = self.conv_d3(d3)  # [B, mid_d, 64, 64]
        d4 = self.conv_d4(d4)  # [B, mid_d, 32, 32]
        d5 = self.conv_d5(d5)  # [B, mid_d, 16, 16]
        
        p5, mask_p5 = self.sam_p5(d5)  # p5: [B, mid_d, 16, 16]
        hs_p5 = F.interpolate(high_semantic, size=p5.shape[2:], mode='bilinear', align_corners=False)
        p5 = self.fusion_block_p5(p5, hs_p5)  # [B, mid_d, 16, 16]

        p4 = self.conv_p4(d4 + F.interpolate(p5, scale_factor=2, mode='bilinear', align_corners=False))  # [B, mid_d, 32, 32]
        p4, mask_p4 = self.sam_p4(p4)
        hs_p4 = F.interpolate(high_semantic, size=p4.shape[2:], mode='bilinear', align_corners=False)
        p4 = self.fusion_block_p4(p4, hs_p4)

        p3 = self.conv_p3(d3 + F.interpolate(p4, scale_factor=2, mode='bilinear', align_corners=False))  # [B, mid_d, 64, 64]
        p3, mask_p3 = self.sam_p3(p3)
        hs_p3 = F.interpolate(high_semantic, size=p3.shape[2:], mode='bilinear', align_corners=False)
        p3 = self.fusion_block_p3(p3, hs_p3)
        

        p2= self.conv_p2(d2 + F.interpolate(p3, scale_factor=2, mode='bilinear', align_corners=False))  # [B, mid_d, 128, 128]
        p2, mask_p2 = self.sam_p2(p2)
        p2_feat_low = self.cat_p2(torch.cat([p2,low_feature],dim=1))
        hs_p2 = F.interpolate(high_semantic, size=p2_feat_low.shape[2:], mode='bilinear', align_corners=False)
        p2_feat = self.fusion_block_p2(p2_feat_low, hs_p2)
        p_low_high = p2_feat

        p2 = self.cls(p2_feat)  # [B, 1, 128, 128]
        p_conv = p2
        p2 = F.interpolate(p2, scale_factor=4, mode='bilinear', align_corners=False)  # [B, 1, 256, 256]
        
        return p2


class BCD(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = resnet18(pretrained=False,progress=False)
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # [batch_size, 64, h/2, w/2]
        self.layer1 = nn.Sequential(backbone.maxpool,backbone.layer1)  # [batch_size, 128, h/4, w/4]
        self.layer2 = backbone.layer2  # [batch_size, 256, h/8, w/8]
        self.layer3 = backbone.layer3  # [batch_size, 512, h/16, w/16]
        
        self.swa = NeighborFeatureAggregation([64, 128, 256, 512])
        
        self.up_2x2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_4x4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.up_8x8 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        self.up_16x16 = nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True)

        self.head=adjust(128,256,512,1024,1)

    def forward(self, t1,t2):
        """ Backbone: ResNet50 """
        # 深度特征提取
        x11 = self.layer0(t1)   ## [-1, 64, h/2, w/2]
        x12 = self.layer1(x11)  ## [-1, 128, h/4, w/4]
        x13 = self.layer2(x12)  ## [-1, 256, h/8, w/8]
        x14 = self.layer3(x13)  ## [-1, 512, h/16, w/16]
        
        x21 = self.layer0(t2)   ## [-1, 64, h/2, w/2]
        x22 = self.layer1(x21)  ## [-1, 128, h/4, w/4]
        x23 = self.layer2(x22)  ## [-1, 256, h/8, w/8]
        x24 = self.layer3(x23)  ## [-1, 512, h/16, w/16]

        s11,s12,s13,s14 = self.swa(x11,x12,x13,x14)
        s21,s22,s23,s24 = self.swa(x21,x22,x23,x24)
        
        u11 = self.up_2x2(s11)  ## [B,64,256,256]
        u12 = self.up_4x4(s12)  ## [B,128,256,256]
        u13 = self.up_8x8(s13)  ## [B,256,256,256]
        u14 = self.up_16x16(s14)## [B,512,256,256]

        u21 = self.up_2x2(s21)  ## [B,64,256,256]
        u22 = self.up_4x4(s22)  ## [B,128,256,256]
        u23 = self.up_8x8(s23)  ## [B,256,256,256]
        u24 = self.up_16x16(s24)## [B,512,256,256]
        
        pred=self.head(u11,u12,u13,u14,u21,u22,u23,u24)

        return s11,s12,s13,s14,s21,s22,s23,s24,u11

class BCD_COLD(nn.Module):
    def __init__(self,checkpoint_path):
        super().__init__()
        """ Backbone: ResNet """
        """从ResNet中提取出layer0, layer1, layer2, layer3"""
        backbone = resnet18()
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # [batch_size, 64, h/2, w/2]
        self.layer1 = nn.Sequential(backbone.maxpool,backbone.layer1)  # [batch_size, 128, h/4, w/4]
        self.layer2 = backbone.layer2  # [batch_size, 256, h/8, w/8]
        self.layer3 = backbone.layer3  # [batch_size, 512, h/16, w/16]

        self.up_2x2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_4x4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.up_8x8 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        self.up_16x16 = nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True)

        self.head=adjust(128,256,512,1024,1)

        self.load_pretrained_weights(checkpoint_path)

    def load_pretrained_weights(self, checkpoint_path):
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.load_state_dict(checkpoint)
            for param in self.parameters():
                param.requires_grad = False
        else:
            print("没有加载预训练权重")

    def forward(self, t1,t2):
        """ Backbone: ResNet34 """
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
        # return x11
        return x11,x12,x13,x14,x21,x22,x23,x24,pred

class EntCD(nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()
        self.mid = 32

        self.cd_stage1cold = BCD_COLD(checkpoint_path)
        self.cd_stage1 = BCD()

        self.CBAM1 = LightweightFusionModule(in_channels=64)
        self.CBAM2 = LightweightFusionModule(in_channels=128)
        self.CBAM3 = LightweightFusionModule(in_channels=256)
        self.CBAM4 = LightweightFusionModule(in_channels=512)


        self.DEGA1 = DualEntropyGuidedAttentionFusion(c=64,h=128)
        self.DEGA2 = DualEntropyGuidedAttentionFusion(c=128,h=64)
        self.DEGA3 = DualEntropyGuidedAttentionFusion(c=256,h=32)
        self.DEGA4 = DualEntropyGuidedAttentionFusion(c=512,h=16)
        

        self.hl_fusion = HighLevelSemanticFusion(in_channels=512, out_channels=256)
        self.lw_fusion = LowLevelGatedFusion(high_channels=256, low_channels=64)

        self.decoder = Decoder(self.mid*2)
        
        self.conv_fuse=nn.Conv2d(4*64, 1, kernel_size=2, padding=0, bias=False)

    def forward(self, t1, t2):

        output_cold = self.cd_stage1cold(t1,t2)

        output = self.cd_stage1(t1,t2)

        # EGASM1_part1:CBAM---entropy-aware feature alignment
        res1 = self.CBAM1(output[0],output[4])
        res2 = self.CBAM2(output[1],output[5])
        res3 = self.CBAM3(output[2],output[6])
        res4 = self.CBAM4(output[3],output[7])
        res1_cold = self.CBAM1(output_cold[0],output_cold[4])
        res2_cold = self.CBAM2(output_cold[1],output_cold[5])
        res3_cold = self.CBAM3(output_cold[2],output_cold[6])
        res4_cold = self.CBAM4(output_cold[3],output_cold[7])

        # EGASM_part2:DEGA---entropy-based strategic selection
        adp1 = self.DEGA1(res1[0],res1_cold[0])
        adp2 = self.DEGA2(res2[0],res2_cold[0])
        adp3 = self.DEGA3(res3[0],res3_cold[0])
        adp4 = self.DEGA4(res4[0],res4_cold[0])

        high_semantic = self.hl_fusion(output[3], output[7]) 
        low_feature = self.lw_fusion(high_semantic, output[0],output[4])

        # fpn
        p = self.decoder(adp1[0], adp2[0], adp3[0], adp4[0],high_semantic, low_feature)
        return p

if __name__ == "__main__":

    from thop import profile
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ConDCD(checkpoint_path='res18_cold_levir.pth').to(device)

    # model.eval()
    t1 = torch.randn(1, 3, 256, 256).to(device)
    t2 = torch.randn(1, 3, 256, 256).to(device)

    # out = model(t1,t2).to(device)

    flops, params = profile(model, inputs=(t1,t1))
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs", end=" | ")
    print(f"Params: {params / 1e6:.2f} M")






