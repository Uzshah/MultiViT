import torch
import torch.nn as nn
import math
from timm.models.registry import register_model
from timm.models.layers import DropPath, trunc_normal_
import torch.nn.functional as F


def stem(in_chs, out_chs):
    """
    Stem Layer that is implemented by two layers of conv.
    Input:
        in_channels: input channels
        out_channels: output channels
    Output: sequence of layers with final shape of [B, C, H/4, W/4]
    """
    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs//2, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs//2),
        nn.GELU(),
        nn.Conv2d(out_chs//2, out_chs, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs),
        nn.GELU(), )


class EleViT_Attn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, padding=3, dilation_rate=1,
                 num_heads=8,bias=True, drop_out = 0.0):
        super(EleViT_Attn, self).__init__()
        self.num_heads = num_heads
        self.head_dim = out_channels
        # Split the input into multiple heads
        self.qkv_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 3, kernel_size, 
                          1, padding),
                nn.BatchNorm2d(out_channels * 3)
            )
        self.apply(self._init_weights)
        self.scale = math.sqrt(self.head_dim)
        self.soft = nn.Softmax(dim = -1)
        self.proj = nn.Sequential(
                nn.GELU(),
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        
        self.focusing_factor = nn.Parameter(torch.zeros(1))
        self.drop = nn.Dropout(drop_out) if drop_out> 0 else nn.Identity()
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
                
    def forward(self, x):
        qkv = self.qkv_conv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        attn = (q @ k.transpose(-1, -2))/self.scale
        attn = self.soft(attn)
        attn = ((self.focusing_factor*attn) @ v)
        out = self.drop(attn)
        out = self.proj(out)
        return out

class ConvEncoder(nn.Module):
    """
    Implementation of ConvEncoder with 3*3 and 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    Output: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_channels, out_channels=64, kernel_size=3, stride=1):
        super().__init__()
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2, groups=in_channels)
        self.norm = nn.BatchNorm2d(in_channels)
        self.pwconv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        

    def forward(self, x):
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.norm2(x)
        return x


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., mid_conv=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mid_conv = mid_conv
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

        if self.mid_conv:
            self.mid = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features)
            self.mid_norm = nn.BatchNorm2d(hidden_features)

        self.norm1 = nn.BatchNorm2d(hidden_features)
        self.norm2 = nn.BatchNorm2d(out_features)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)

        if self.mid_conv:
            x_mid = self.mid(x)
            x_mid = self.mid_norm(x_mid)
            x = self.act(x_mid)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.norm2(x)

        x = self.drop(x)
        return x

class Attention_block(nn.Module):
    def __init__(self, dim, drop_path=0., mlp_ratio = 4):
        super().__init__()
        self.conv = ConvEncoder(dim, dim)
        self.attn = EleViT_Attn(dim, dim, kernel_size = 3, padding=1, stride= 1, num_heads = 1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.mlp = Mlp(in_features = dim, hidden_features = dim*mlp_ratio, mid_conv = True)
        
    def forward(self, x):
        x = x + self.drop_path(self.conv(x))
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))
        return x

class Downsampling(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layer = nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, 1, stride = 2),
                    nn.BatchNorm2d(out_dim),
                    nn.GELU())
    def forward(self, x):
        return self.layer(x)

class Upsampling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )

    def forward(self, x): 
        return self.deconv(x)

class AConv(nn.Module):
    def __init__(self, in_c, out_c = None, ks=3, st=(1, 1)):
        super(AConv, self).__init__()
        out_c = out_c or in_c
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=ks, stride=st, padding=ks//2),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
        )

    def forward(self, x):
        return self.layers(x)
        
class head(nn.Module):
    def __init__(self, num_classes, lfeat, mode = "nearest"):
        super(head, self).__init__()
        self.mode = mode
        self.decoder = nn.ModuleList([AConv(lfeat, lfeat//2),
                                     AConv(lfeat//2, lfeat//2)])
        self.output = nn.Conv2d(lfeat//2, num_classes, kernel_size=3, padding = 1)
    def forward(self, x):
        for conv in self.decoder:
            x = F.interpolate(x, scale_factor=(2,2), mode=self.mode)
            x = conv(x)
            
        out = self.output(x)
        return out
        
class MultiViT(nn.Module):
    def __init__(self, num_classes, target, d= [64, 96, 128, 192, 256]):
        super().__init__()
        self.target = target
        self.input  = stem(3, d[0])
        if self.target=="all" or self.target == "depth":
            self.depth = head(1, d[0]*2)
        if self.target=="all" or self.target == "shading":
            self.shading = head(1, d[0]*2)
        if self.target=="all" or self.target == "normal":
            self.normal = head(3, d[0]*2)
        if self.target=="all" or self.target == "albedo":
            self.albedo = head(3, d[0]*2)
        if self.target=="all" or self.target == "semantic":
            self.semantic = head(num_classes, d[0]*2)


        self.encode_layer1 = Attention_block(d[0])
        self.downsampling1 = Downsampling(d[0], d[1])

        self.encode_layer2 = Attention_block(d[1])
        self.downsampling2 = Downsampling(d[1], d[2])
        
        self.encode_layer3 = Attention_block(d[2])
        self.downsampling3= Downsampling(d[2], d[3])

        self.encode_layer4 = Attention_block(d[3])
        self.downsampling4= Downsampling(d[3], d[4])
        
        self.bottleneck = Attention_block(d[4])

        self.upsampling4 = Upsampling(d[4], d[3])
        self.decoder_layer4 = Attention_block(d[3]*2)
        
        self.upsampling3 = Upsampling(d[3]*2, d[2])
        self.decoder_layer3 = Attention_block(d[2]*2)

        self.upsampling2 = Upsampling(d[2]*2, d[1])
        self.decoder_layer2 = Attention_block(d[1]*2)
        
        self.upsampling1 = Upsampling(d[1]*2, d[0])
        self.decoder_layer1 = Attention_block(d[0]*2)

    def forward(self, x):
        x = self.input(x)
        layer1 = self.encode_layer1(x)
        pool = self.downsampling1(layer1)
        layer2 = self.encode_layer2(pool)
        pool = self.downsampling2(layer2)
        layer3 = self.encode_layer3(pool)
        pool = self.downsampling3(layer3)
        layer4 = self.encode_layer4(pool)
        pool = self.downsampling4(layer4)

        bot = self.bottleneck(pool)
        
        up4 = self.upsampling4(bot)
        dlayer4 = torch.cat([up4, layer4], dim =1)
        # print(dlayer4.size())
        dlayer4 = self.decoder_layer4(dlayer4)
        
        up3 = self.upsampling3(dlayer4)
        dlayer3 = torch.cat([up3, layer3], dim =1)
        # print(dlayer3.size())
        dlayer3 = self.decoder_layer3(dlayer3)
        
        up2 = self.upsampling2(dlayer3)
        dlayer2 = torch.cat([up2, layer2], dim =1)
        # print(dlayer2.size())
        dlayer2 = self.decoder_layer2(dlayer2)
        
        up1 = self.upsampling1(dlayer2)
        dlayer1 = torch.cat([up1, layer1], dim =1)
        dlayer1 = self.decoder_layer1(dlayer1)
        # print(dlayer1.size())
        
        outputs = {}
        if self.target=="all" or self.target == "depth":
            outputs["pred_depth"] = self.depth(dlayer1)
        if self.target=="all" or self.target == "shading":
            outputs["pred_shading"] = self.shading(dlayer1)
        if self.target=="all" or self.target == "normal":
            outputs["pred_normal"] = self.normal(dlayer1)
        if self.target=="all" or self.target == "albedo":
            outputs["pred_albedo"] = self.albedo(dlayer1)
        if self.target=="all" or self.target == "semantic":
            outputs["pred_semantic"] = self.semantic(dlayer1)
        return outputs