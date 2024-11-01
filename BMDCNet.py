import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from functools import partial

nonlinearity = partial(F.relu, inplace=True)


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=3, stride=2, padding=1, in_chans=3, embed_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, patch_size, stride=stride, padding=padding),
            nn.BatchNorm2d(embed_dim)
        )

    def forward(self, x):
        return self.proj(x)


class ECA(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class MultiScaleDWConv(nn.Module):
    def __init__(self, dim, scale=(1, 3, 5, 7)):
        super().__init__()
        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
            else:
                channels = dim // len(scale)
            conv = nn.Conv2d(channels, channels, kernel_size=scale[i], padding=scale[i] // 2, groups=channels)
            self.channels.append(channels)
            self.proj.append(conv)

    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(hidden_features),
        )
        self.dwconv = MultiScaleDWConv(hidden_features)
        self.act = nn.GELU()
        self.norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_features),
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x) + x
        x = self.norm(self.act(x))
        x = self.fc2(x)
        return x


class StripDynamicConv2d(nn.Module):
    def __init__(self, dim, idx=-1, kernel_size=3, reduction_ratio=4, num_groups=4, dilation=1, bias=True):
        super().__init__()
        assert num_groups > 1, f"num_groups {num_groups} should > 1."
        self.idx = idx
        self.dilation = dilation
        self.num_groups = num_groups
        self.K = kernel_size
        if self.idx == 0:
            self.weight = nn.Parameter(torch.empty(num_groups, dim, kernel_size, 1), requires_grad=True)
            self.pool = nn.AdaptiveAvgPool2d(output_size=(kernel_size, 1))
        elif self.idx == 1:
            self.weight = nn.Parameter(torch.empty(num_groups, dim, 1, kernel_size), requires_grad=True)
            self.pool = nn.AdaptiveAvgPool2d(output_size=(1, kernel_size))
        elif self.idx == -1:
            self.weight = nn.Parameter(torch.empty(num_groups, dim, kernel_size, kernel_size), requires_grad=True)
            self.pool = nn.AdaptiveAvgPool2d(output_size=(kernel_size, kernel_size))
        else:
            print("ERROR MODE", self.idx)
            exit(0)
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim // reduction_ratio, kernel_size=1),
            nn.BatchNorm2d(dim // reduction_ratio),
            nn.GELU(),
            nn.Conv2d(dim // reduction_ratio, dim * num_groups, kernel_size=1)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(num_groups, dim), requires_grad=True)
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        if self.idx == 0:
            scale = self.proj(self.pool(x)).reshape(B, self.num_groups, C, self.K, 1)
        elif self.idx == 1:
            scale = self.proj(self.pool(x)).reshape(B, self.num_groups, C, 1, self.K)
        else:
            scale = self.proj(self.pool(x)).reshape(B, self.num_groups, C, self.K, self.K)
        scale = torch.softmax(scale, dim=1)
        weight = scale * self.weight.unsqueeze(0)
        weight = torch.sum(weight, dim=1, keepdim=False)
        if self.idx == 0:
            weight = weight.reshape(-1, 1, self.K, 1)
            if self.dilation == 1:
                padding = (self.K // 2, 0)
            else:
                padding = (self.dilation, 0)
        elif self.idx == 1:
            weight = weight.reshape(-1, 1, 1, self.K)
            if self.dilation == 1:
                padding = (0, self.K // 2)
            else:
                padding = (0, self.dilation)
        else:
            weight = weight.reshape(-1, 1, self.K, self.K)
            if self.dilation == 1:
                padding = self.K // 2
            else:
                padding = self.dilation
        if self.bias is not None:
            scale = self.proj(torch.mean(x, dim=[-2, -1], keepdim=True))
            scale = torch.softmax(scale.reshape(B, self.num_groups, C), dim=1)
            bias = scale * self.bias.unsqueeze(0)
            bias = torch.sum(bias, dim=1).flatten(0)
        else:
            bias = None
        x = F.conv2d(x.reshape(1, -1, H, W), weight=weight, bias=bias, padding=padding, dilation=self.dilation,
                     groups=B * C)
        return x.reshape(B, C, H, W)


class EDA_Attention(nn.Module):
    def __init__(self, dim, idx, split_size=8, num_heads=8, qk_scale=None):
        super().__init__()
        self.idx = idx
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.split_size = split_size
        self.scale = qk_scale or head_dim ** -0.5
        self.pos = StripDynamicConv2d(dim)
        self.softmax = nn.Softmax(dim=-1)

    def im2cswin(self, x, H, W, H_sp, W_sp):
        B, _, C = x.shape
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
        x = x.reshape(-1, H_sp * W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_v(self, x, H, W, H_sp, W_sp):
        B, C, _, _ = x.shape
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)
        x = x.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()
        return x

    def forward(self, q, k, v, H, W):
        if self.idx == 0:
            H_sp, W_sp = H, self.split_size
        elif self.idx == 1:
            W_sp, H_sp = W, self.split_size
        else:
            print("ERROR MODE", self.idx)
            exit(0)
        B, _, C = q.shape
        q = self.im2cswin(q, H, W, H_sp, W_sp)
        k = self.im2cswin(k, H, W, H_sp, W_sp)
        v = v.transpose(-2, -1).contiguous().view(B, C, H, W)
        pos = self.pos(v)
        v = self.get_v(v, H, W, H_sp, W_sp)
        pos = self.get_v(pos, H, W, H_sp, W_sp)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)
        x = (attn @ v) + pos
        x = x.transpose(1, 2).reshape(-1, H_sp * W_sp, C)
        x = x.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, -1, C)
        return x


class EDA_Block(nn.Module):
    def __init__(self, dim, num_heads=4, split_size=8, qk_scale=None,
                 mlp_ratio=4):  # , drop=0, drop_path=0,layer_scale_init_value=1e-5
        super().__init__()
        assert dim % 2 == 0, f"dim {dim} should be divided by 2."
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.pos_embed = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm1 = nn.GroupNorm(1, dim)
        self.qkv = nn.Linear(dim, 5 * dim)
        self.attn_x = EDA_Attention(dim, idx=0, split_size=split_size, num_heads=num_heads, qk_scale=qk_scale)
        self.attn_y = EDA_Attention(dim, idx=1, split_size=split_size, num_heads=num_heads, qk_scale=qk_scale)
        self.proj = nn.Sequential(
            nn.Conv2d(2 * dim, dim, kernel_size=1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.BatchNorm2d(dim),
            ECA(dim)
        )
        self.norm2 = nn.GroupNorm(1, dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x + self.pos_embed(x)
        shorcut = x.clone()
        x = self.norm1(x).view(B, C, -1).permute(0, 2, 1).contiguous()
        x = self.qkv(x)
        qkv = []
        for i in range(5):
            qkv.append(x[:, :, i::5])
        x1 = self.attn_x(qkv[0], qkv[1], qkv[2], H, W).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x2 = self.attn_y(qkv[3], qkv[4], qkv[2], H, W).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x = torch.cat([x1, x2], dim=1)
        x = self.proj(x) + shorcut
        x = self.mlp(self.norm2(x)) + x
        return x


class SID_DA(nn.Module):
    def __init__(self, dim):
        super(SID_DA, self).__init__()
        blocks = nn.ModuleList()
        blocks.append(EDA_Block(dim, split_size=2))
        blocks.append(EDA_Block(dim, split_size=4))
        blocks.append(EDA_Block(dim, split_size=8))
        blocks.append(EDA_Block(dim, split_size=32))
        self.convs = blocks
        self.proj = nn.Sequential(
            ECA(5 * dim),
            nn.Conv2d(5 * dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        shorcut = x.clone()
        res = [x]
        for conv in self.convs:
            x = conv(x)
            res.append(x)
        res = torch.cat(res, dim=1)
        x = self.proj(res) + shorcut
        return x


class SDC_ASPPConv(nn.Sequential):
    def __init__(self, dim, idx=-1, dilation=1):
        modules = [
            StripDynamicConv2d(dim, idx=idx, kernel_size=3, dilation=dilation),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        ]
        super(SDC_ASPPConv, self).__init__(*modules)


class SID_ASPP(nn.Module):
    def __init__(self, dim):
        super(SID_ASPP, self).__init__()
        blocks = nn.ModuleList()
        blocks.append(nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)))
        blocks.append(SDC_ASPPConv(dim, idx=-1))
        blocks.append(SDC_ASPPConv(dim, idx=0))
        blocks.append(SDC_ASPPConv(dim, idx=1))
        blocks.append(SDC_ASPPConv(dim, idx=-1, dilation=3))
        blocks.append(SDC_ASPPConv(dim, idx=0, dilation=3))
        blocks.append(SDC_ASPPConv(dim, idx=1, dilation=3))
        blocks.append(SDC_ASPPConv(dim, idx=-1, dilation=6))
        blocks.append(SDC_ASPPConv(dim, idx=0, dilation=6))
        blocks.append(SDC_ASPPConv(dim, idx=1, dilation=6))
        blocks.append(SDC_ASPPConv(dim, idx=-1, dilation=12))
        blocks.append(SDC_ASPPConv(dim, idx=0, dilation=12))
        blocks.append(SDC_ASPPConv(dim, idx=1, dilation=12))
        blocks.append(ASPPPooling(dim, dim))
        self.convs = blocks
        self.proj = nn.Sequential(
            ECA(15 * dim),
            nn.Conv2d(15 * dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True))

    def forward(self, x):
        shorcut = x.clone()
        res = [x]
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        x = self.proj(res) + shorcut
        # x = self.mlp(self.norm(res)) + x
        return x


class DCDE(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        assert dim % 2 == 0, f"dim {dim} should be divided by 2."
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.pos_embed = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm1 = nn.GroupNorm(1, dim)
        self.local_unit = SID_ASPP(dim // 2)
        self.global_unit = SID_DA(dim // 2)
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.BatchNorm2d(dim),
            ECA(dim)
        )
        self.norm2 = nn.GroupNorm(1, dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x1, x2 = torch.chunk(self.norm1(x), chunks=2, dim=1)
        x1 = self.local_unit(x1)
        x2 = self.global_unit(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.proj(x) + x
        x = self.mlp(self.norm2(x)) + x
        return x


class SLSA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv1_0 = nn.Conv2d(dim // 4, dim // 4, (1, 3), padding=(0, 1), groups=dim // 4)
        self.conv1_1 = nn.Conv2d(dim // 4, dim // 4, (3, 1), padding=(1, 0), groups=dim // 4)
        self.conv2_0 = nn.Conv2d(dim // 4, dim // 4, (1, 7), padding=(0, 3), groups=dim // 4)
        self.conv2_1 = nn.Conv2d(dim // 4, dim // 4, (7, 1), padding=(3, 0), groups=dim // 4)
        self.conv3_0 = nn.Conv2d(dim // 4, dim // 4, (1, 7), padding=(0, 6), groups=dim // 4, dilation=2)
        self.conv3_1 = nn.Conv2d(dim // 4, dim // 4, (7, 1), padding=(6, 0), groups=dim // 4, dilation=2)
        self.conv4_0 = nn.Conv2d(dim // 4, dim // 4, (1, 7), padding=(0, 9), groups=dim // 4, dilation=3)
        self.conv4_1 = nn.Conv2d(dim // 4, dim // 4, (7, 1), padding=(9, 0), groups=dim // 4, dilation=3)
        self.conv5 = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv6 = nn.Conv2d(3 * dim, dim, 1)
        self.conv_squeeze_x = nn.Conv2d(dim, 1, 1)
        self.conv_squeeze_y = nn.Conv2d(dim, 1, 1)
        self.conv_squeeze = nn.Conv2d(4, 4, 7, padding=3)
        self.conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        attn = self.conv0(x)
        attn1_1 = self.conv1_0(attn[:, 0::4, :, :]) + self.conv1_1(attn[:, 0::4, :, :])
        attn1_2 = self.conv2_0(attn[:, 1::4, :, :]) + self.conv2_1(attn[:, 1::4, :, :])
        attn1_3 = self.conv3_0(attn[:, 2::4, :, :]) + self.conv3_1(attn[:, 2::4, :, :])
        attn1_4 = self.conv4_0(attn[:, 3::4, :, :]) + self.conv4_1(attn[:, 3::4, :, :])
        attn1 = torch.cat([attn1_1, attn1_2, attn1_3, attn1_4], dim=1)
        attn2 = self.conv5(attn)
        attn = self.conv6(torch.cat([attn, attn1, attn2], dim=1))
        attn_x = self.conv_squeeze_x(F.avg_pool2d(attn, (attn.size(2), 1), stride=(attn.size(2), 1)).expand_as(x))
        attn_y = self.conv_squeeze_y(F.avg_pool2d(attn, (1, attn.size(3)), stride=(1, attn.size(3))).expand_as(x))
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        sig = self.conv_squeeze(torch.cat([avg_attn, max_attn, attn_x, attn_y], dim=1)).sigmoid()
        spatial_attn = ((sig[:, 0, :, :] + sig[:, 1, :, :] + sig[:, 2, :, :] + sig[:, 3, :, :]) / 4.0).unsqueeze(1)
        attn = attn * spatial_attn
        attn = self.conv(attn)
        return x * attn


class DF(nn.Module):
    def __init__(self, dim, mode='down', mlp_ratio=4):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        if mode == 'up':
            self.patch_y = nn.ConvTranspose2d(2 * dim, dim, 3, stride=2, padding=1, output_padding=1)
        elif mode == 'down':
            self.patch_y = PatchEmbed(patch_size=7, stride=2, padding=3, in_chans=dim // 2, embed_dim=dim)
        else:
            print("ERROR MODE", mode)
            exit(0)
        # self.norm1 = nn.BatchNorm2d(dim)
        self.norm1 = nn.GroupNorm(1, dim)
        self.act = nn.GELU()
        self.attention = nn.Sequential(
            SLSA(dim),
            ECA(dim)
        )
        self.norm2 = nn.GroupNorm(1, dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x, y):
        shorcut = x.clone()
        y = self.act(self.norm1(self.patch_y(y)))
        x = self.attention(0.75 * x + 0.25 * y) + shorcut
        x = self.mlp(self.norm2(x)) + x
        return x


class BMDF(nn.Module):
    def __init__(self, dims):
        super(BMDF, self).__init__()
        self.norm1 = nn.BatchNorm2d(dims[0])
        self.norm2 = nn.BatchNorm2d(dims[1])
        self.norm3 = nn.BatchNorm2d(dims[2])
        self.norm4 = nn.BatchNorm2d(dims[3])
        self.down1 = DF(dims[1], mode='down', mlp_ratio=2)
        self.down2 = DF(dims[2], mode='down', mlp_ratio=2)
        self.down3 = DF(dims[3], mode='down')
        self.up3 = DF(dims[2], mode='up')
        self.up2 = DF(dims[1], mode='up', mlp_ratio=2)
        self.up1 = DF(dims[0], mode='up', mlp_ratio=2)

    def forward(self, e1, e2, e3, e4):
        ne1 = self.norm1(e1)
        ne2 = self.norm2(e2)
        ne3 = self.norm3(e3)
        ne4 = self.norm4(e4)
        # downsample fuse phase
        de2 = self.down1(ne2, ne1)
        de3 = self.down2(ne3, de2)
        de4 = self.down3(ne4, de3)
        # upsample fuse phase
        ue3 = self.up3(de3, de4)
        ue2 = self.up2(de2, ue3)
        ue1 = self.up1(ne1, ue2)
        e1 = ue1 + e1
        e2 = ue2 + e2
        e3 = ue3 + e3
        e4 = de4 + e4
        return e1, e2, e3, e4


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class BMDCNet(nn.Module):
    def __init__(self, img_size=1024, num_classes=1):
        super(BMDCNet, self).__init__()
        self.img_size=img_size
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.skip = BMDF(filters)
        self.dblock = DCDE(filters[3])

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        if self.img_size != 1024:
            x = F.interpolate(x, size=1024, mode='bilinear', align_corners=True)
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        # Fussion
        e1, e2, e3, e4 = self.skip(e1, e2, e3, e4)
        # Center
        e4 = self.dblock(e4)
        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        # SegHead
        if self.img_size != 1024:
            out = F.interpolate(out, size=self.img_size, mode='bilinear', align_corners=True)
        return out


if __name__ == '__main__':
    from torchinfo import summary
    from ptflops import get_model_complexity_info

    net = BMDCNet(img_size=1500)
    summary(net, input_size=(4, 3, 1500, 1500))
    macs, params = get_model_complexity_info(net, (3, 1500, 1500), print_per_layer_stat=False)
    print(macs, params)
