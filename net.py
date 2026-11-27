import torch.nn.init as init
from torch import Tensor
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange
import numpy as np
import numbers

import torch
import torch.nn.functional as F
import torch.nn as nn
from functools import partial
import pywt
import pywt.data


class Conv1(nn.Module):
    """1x1 卷积，保持风格一致"""
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)

    def forward(self, x):
        return self.conv(x)


DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


def drop_path(x, drop_prob: float = 0., training: bool = False):

    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
                    torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)



class Pooling(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        return self.pool(x) 

class PoolMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features    = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1  = nn.Conv2d(in_features,  hidden_features, 1, bias=False)
        self.act  = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2  = nn.Conv2d(hidden_features, out_features, 1, bias=False)

    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x


class BaseFeatureExtraction(nn.Module):
    def __init__(self, dim, pool_size=5, mlp_ratio=4., act_layer=nn.GELU,
                 drop=0., drop_path=0., use_layer_scale=False, layer_scale_init_value=1e-5):
        super().__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.token_mixer = nn.Sequential(
            Pooling(kernel_size=pool_size),           
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim), 
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False)  
        )
        self.norm2 = LayerNorm(dim, 'WithBias')
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.poolmlp = PoolMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(torch.ones(dim, dtype=torch.float32) * layer_scale_init_value)
            self.layer_scale_2 = nn.Parameter(torch.ones(dim, dtype=torch.float32) * layer_scale_init_value)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.poolmlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.poolmlp(self.norm2(x)))
        return x


class Sobelxy(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))

    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x = torch.abs(sobelx) + torch.abs(sobely)
        return x

class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out

        return self.sigmoid(out)


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SFIM(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4, kernel_size=7):
        super(SFIM, self).__init__()
        self.channel_attention = ChannelAttentionModule(in_channels, reduction)
        self.spatial_attention = SpatialAttentionModule(kernel_size)

  
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.refine_fre = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.refine_spa = nn.Conv2d(64, 64, kernel_size=1, bias=False)

        self.up = Conv1(in_channels, out_channels)

    def forward(self, fre, spa):
        sub_input = fre - spa                    
        conv_sub  = self.conv(sub_input)         

        x_out_sam = self.spatial_attention(conv_sub)   

        x_out_spa = (sub_input * (1 - x_out_sam)) + spa
        x_out_fre = (sub_input * x_out_sam)     + fre

        x_out_spa = self.refine_spa(x_out_spa)   
        x_out_fre = self.refine_fre(x_out_fre)   

        x_out_concat = torch.cat((x_out_fre, x_out_spa), dim=1)  
        x_out = self.channel_attention(x_out_concat)             

        return x_out

class FCA(nn.Module):

    def __init__(self, out_channel, norm='backward'):
        super(FCA, self).__init__()
        self.norm = norm
        hidden = max(4, out_channel // 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(out_channel, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channel, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.act = nn.Sigmoid()
        self.eps = 1e-6

    @torch.cuda.amp.autocast(enabled=False)  
    def forward(self, x):
        x32 = x.to(torch.float32)
        X = torch.fft.fft2(x32, norm=self.norm)         

        A = torch.abs(X).clamp_min(self.eps)
        a = A.mean(dim=(-2, -1), keepdim=True)         

        w = self.mlp(a)                                 

        Xw = X * w                                   

        y_hat = torch.fft.ifft2(Xw, norm=self.norm).real 

        y = self.act(y_hat.to(x.dtype)) * x
        return y



class Frequency_Domain(nn.Module):
    def __init__(self, in_c, out_c, factor=4.0):
        super().__init__()
        self.out_c = out_c

        self.fam = FCA(out_channel=out_c)

    def forward(self, x):
        output_fam = self.fam(x)
        return output_fam
        # return x


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.1, use_HIN=True):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        out = self.conv_1(x)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=16, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = UNetConvBlock(channel_in, gc)
        self.conv2 = UNetConvBlock(gc, channel_out)
        # self.conv3 = nn.Conv2d(channel_in + 2 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2], 0.1)
        # initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        # x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))

        return x2


class INN(nn.Module):
    def __init__(self, subnet_constructor=DenseBlock, clamp=2.0, harr=True, in_1=16, in_2=16):
        super().__init__()
        if harr:
            self.split_len1 = in_1 * 2
            self.split_len2 = in_2 * 2
        self.clamp = clamp

        # ρ
        self.r = subnet_constructor(self.split_len1, self.split_len2)
        # η
        self.y = subnet_constructor(self.split_len1, self.split_len2)
        # φ
        self.f = subnet_constructor(self.split_len2, self.split_len1)
        # ψ
        self.p = subnet_constructor(self.split_len2, self.split_len1)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x):
        x1, x2 = (x.narrow(1, 0, self.split_len1),
                  x.narrow(1, self.split_len1, self.split_len2))

        t2 = self.f(x2)
        s2 = self.p(x2)
        y1 = self.e(s2) * x1 + t2
        s1, t1 = self.r(y1), self.y(y1)
        y2 = self.e(s1) * x2 + t1

        return torch.cat((y1, y2), 1)


class DetailFeatureExtraction(nn.Module):
    def __init__(self, in_c, out_c, factor=4.0):
        super().__init__()
        self.out_c = out_c
        self.fre = Frequency_Domain(in_c=64, out_c=64)
        self.inn = INN()
        self.interaction = SFIM(in_channels=in_c * 2, out_channels=out_c)
        self.Sobel_xy = Sobelxy(in_c)
        self.down1 = Conv1(in_channels=in_c * 2, out_channels=out_c)

    def forward(self, x):
        output_embed =x
        output_fre   = self.fre(output_embed)     
        output_spa   = self.inn(output_embed)     
        output_inter = self.interaction(output_fre, output_spa)
        output_inter = self.down1(output_inter)               
        output_sobel = self.Sobel_xy(x)                        
        output = output_inter + output_sobel                  
        return output


# =============================================================================
# Layer Norm
# =============================================================================
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):  # (b,c,h,w)
        h, w = x.shape[-2:]
        """
        to_3d后：(b,h*w,c)
        body后：(b,h*w,c)
        to_4d后：(b,c,h,w)
        """
        return to_4d(self.body(to_3d(x)), h, w)


# =============================================================================
# Gated-Dconv Feed-Forward Network (GDFN)
# =============================================================================

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


# =============================================================================
# Multi-DConv Head Transposed Self-Attention (MDTA)
# =============================================================================

class Attention1(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention1, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        qkv = self.qkv_dwconv(qkv)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention1(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # LN->MDTA->残差连接
        x = x + self.ffn(self.norm2(x))  # LN->GDFN->残差连接
        return x


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batch_size, height, width, num_channels = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batch_size, height, width, groups, channels_per_group)
    x = torch.transpose(x, 3, 4).contiguous()
    x = x.view(batch_size, height, width, -1)
    return x


class eca_layer(nn.Module):
    def __init__(self, channel, k_size):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.k_size = k_size
        self.conv = nn.Conv1d(channel, channel, kernel_size=k_size, bias=False, groups=channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = nn.functional.unfold(y.transpose(-1, -3), kernel_size=(1, self.k_size), padding=(0, (self.k_size - 1) // 2))
        y = self.conv(y.transpose(-1, -2)).unsqueeze(-1)
        y = self.sigmoid(y)
        x = x * y.expand_as(x)
        return x

class CorrCoef(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        # x,y: (B,C,H,W)
        mu_x = x.mean(dim=1, keepdim=True)     # (B,1,H,W)
        mu_y = y.mean(dim=1, keepdim=True)
        xc = x - mu_x
        yc = y - mu_y
        vx = (xc * xc).sum(dim=1, keepdim=True).clamp_min(self.eps).sqrt()  # L2范数当作标准差的等价尺度
        vy = (yc * yc).sum(dim=1, keepdim=True).clamp_min(self.eps).sqrt()
        ncc = (xc * yc).sum(dim=1, keepdim=True) / (vx * vy + self.eps)     # (B,1,H,W), ∈[-1,1]
        return ncc



class CorrelationFusionHead(nn.Module):

    def __init__(self, dim, use_eca=True, tau=2.0, reduction=4):
        super().__init__()
        self.dim = dim
        self.ncc = CorrCoef()
        self.tau = tau
        hid = max(dim // reduction, 4)

        self.pre_low  = nn.Conv2d(dim, dim, 1, bias=False)
        self.pre_high = nn.Conv2d(dim, dim, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        if use_eca:
            self.recal = eca_layer(channel=dim * 2, k_size=3)  
        else:
            self.recal = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim * 2, hid, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(hid, dim * 2, 1, bias=False),
                nn.Sigmoid()
            )

        self.post = nn.Conv2d(dim * 2, dim * 2, 1, bias=False)

    def forward(self, low, high):
        low  = self.pre_low(low)
        high = self.pre_high(high)

        r = self.ncc(low, high)               
        g = self.sigmoid(self.tau * r)       

        low_hat  = (1.0 - g) * low
        high_hat = g * high

        cat = torch.cat([low_hat, high_hat], dim=1)  

        w = self.recal(cat)                   
        out = self.post(cat * w) + cat         

        return out

# =============================================================================
# Overlapped image patch embedding with 3x3 Conv
# =============================================================================

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class Restormer_Encoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(Restormer_Encoder, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.baseFeature = BaseFeatureExtraction(dim=dim)
        self.detailFeature = DetailFeatureExtraction(in_c=64, out_c=64)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        base_feature = self.baseFeature(out_enc_level1)
        detail_feature = self.detailFeature(out_enc_level1)

        return base_feature, detail_feature, out_enc_level1


class Restormer_Decoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(Restormer_Decoder, self).__init__()

        self.reduce_channel1 = CorrelationFusionHead(dim=64, use_eca=True, tau=2.0)

        self.reduce_channel2 = nn.Conv2d(in_channels=int(dim * 2), out_channels=int(dim), kernel_size=1, bias=bias)

        self.encoder_level2 = nn.Sequential(
            *[TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=int(dim), out_channels=int(dim) // 2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=int(dim) // 2, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                      bias=bias), )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp_img, base_feature, detail_feature, fuse='MIFA_block'):
        if fuse == 'MIFA_block':
            out_enc_level0 = self.reduce_channel1(base_feature, detail_feature)
            out_enc_level0 = self.reduce_channel2(out_enc_level0)
        else:
            out_enc_level0 = torch.cat((base_feature, detail_feature), dim=1)
            out_enc_level0 = self.reduce_channel2(out_enc_level0)

        out_enc_level1 = self.encoder_level2(out_enc_level0)

        if inp_img is not None:
            out_enc_level1 = self.output(out_enc_level1) + inp_img
        else:
            out_enc_level1 = self.output(out_enc_level1)
        return self.sigmoid(out_enc_level1), out_enc_level0


if __name__ == "__main__":
    import torch
    x = torch.randn(2, 64, 128, 128)
    bfe = BaseFeatureExtraction(dim=64)  
    y = bfe(x)
    print(y.shape) 