from torch import nn
from torchvision.models import efficientnet_b0
import torch
from torch.nn import functional as F
from einops import rearrange

from utils import cls_decode, seg_decode

class SepConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channel,
            in_channel,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=in_channel,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.conv2 = nn.Conv2d(
            in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=bias
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out


class AllinOneModel(nn.Module):
    def __init__(self, nc_det=10, nc_seg=20, nc_cls=10, n_bbox=2):
        super(AllinOneModel, self).__init__()
        self.nc_det = nc_det
        self.n_bbox = n_bbox
        self.nc_seg = nc_seg
        self.nc_cls = nc_cls
        self.bock_dim = max(nc_det+(n_bbox*(4+1)), nc_seg, nc_cls)
        ### backbone ###
        self.backbone = efficientnet_b0(pretrained=True).features
        
        ### neck ###
        self.up = PixelShuffle(2, 1280) # B, 1280, H//32, W//32 -> B, 640, H//16, W//16
        self.neck_cls = nn.Sequential(
            SepConv(640, 128, 3, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.neck_seg = nn.Sequential(
            SepConv(640, 128, 3, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.neck_det = nn.Sequential(
            SepConv(640, 128, 3, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        ### head ###
        # det: (2*(4+1)+10)*H_*W_; seg: 20*H_*W_; cls: 10*H_*W_; => 50*H_*W_
        self.head = nn.Sequential(
            Attention(128),
            nn.Conv2d(128, 64, 3, padding=1, bias=False, groups=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.bock_dim*4, 1, bias=False, groups=4),
        )
        self.idt_cls = nn.Conv2d(128, nc_cls, 1, bias=True)
        self.idt_seg = nn.Conv2d(128, nc_seg, 1, bias=True)
        self.idt_det = nn.Conv2d(128, nc_det+(n_bbox*(4+1)), 1, bias=True)

    def forward(self, x):
        feat = self.backbone(x) # B, 1280, H//32, W//32
        
        # feat = F.interpolate(feat, scale_factor=2, mode="bilinear") # B, 1280, H//16, W//16
        feat = self.up(feat) # B, 512, H//16, W//16
        # feat = self.conv(feat) # B, 512, H//16, W//16
        # feat = self.relu(self.bn(feat))
        feat_cls_ = self.neck_cls(feat) # B, 512, H//16, W//16
        feat_seg_ = self.neck_seg(feat) # B, 512, H//16, W//16
        feat_det_ = self.neck_det(feat) # B, 512, H//16, W//16
        feat = feat_cls_ + feat_seg_ + feat_det_ # B, 512, H//16, W//16
        
        feat = self.head(feat)
        
        feat_det = feat[:, :(self.nc_det+(self.n_bbox*(4+1))), :, :] ## detection output
        feat_seg = feat[:, self.bock_dim:self.bock_dim+self.nc_seg, :, :] ## segmentation output
        feat_cls = feat[:, 2*self.bock_dim:2*self.bock_dim+self.nc_cls, :, :] ## classification output
        ## remaining part is bock_dim channels are redundant
        outs = {
            "det": torch.sigmoid(feat_det + self.idt_det(feat_det_)),
            "seg": seg_decode(feat_seg + self.idt_seg(feat_seg_)),
            "cls": cls_decode(feat_cls + self.idt_cls(feat_cls_)), 
        }
        return outs

class PixelShuffle(nn.Module):
    def __init__(self, upscale_factor, in_channels):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.conv = nn.Conv2d(in_channels, in_channels*upscale_factor, kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        
    def forward(self, x):
        # x (B, C, H, W)
        x = self.conv(x) # (B, C*upscale_factor, H, W)
        x = self.pixel_shuffle(x) # (B, C//upscale_factor, H*upscale_factor, W*upscale_factor)
        return x
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
    
    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b c h w -> b (h w) c'), qkv)
        q = q * self.scale
        sim = torch.einsum('b i d, b j d -> b i j', q, k)
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)
        return self.to_out(out) + x

