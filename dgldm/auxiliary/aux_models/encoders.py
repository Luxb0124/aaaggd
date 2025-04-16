import os
import sys
import math
import torch
from torch import nn
import torch.nn.init as init
sys.path.append(os.path.dirname(__file__))
from blocks import ResBlocks
from torchvision.ops import DeformConv2d


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return init_fun


class ModulatedDeformConvPack(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1,
                 deformable_groups=1, bias=True, lr_mult=0.1):
        super(ModulatedDeformConvPack, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.deformable_groups = deformable_groups

        out_channels = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(self.in_channels, out_channels, kernel_size=self.kernel_size,
                                          stride=self.stride, padding=self.padding, bias=True)
        self.dcn = DeformConv2d(in_channels=in_channels, out_channels=self.out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding,
                                dilation=dilation, groups=groups, bias=bias)

        self.conv_offset_mask.lr_mult = lr_mult
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input_offset, input_real):
        out = self.conv_offset_mask(input_offset)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        out = self.dcn(input_real, offset, mask)
        return out, offset


class ContentEncoder(nn.Module):
    def __init__(self, nf_cnt=64, n_downs=2, n_res=2, norm='in', act='relu', pad='reflect', use_sn=False):
        super(ContentEncoder, self).__init__()
        print("Init ContentEncoder")

        nf = nf_cnt

        self.model = nn.ModuleList()
        self.model.append(ResBlocks(n_res, 256, norm=norm, act=act, pad_type=pad, use_sn=use_sn))
        self.model = nn.Sequential(*self.model)
        self.dcn1 = ModulatedDeformConvPack(3, 64, kernel_size=(7, 7), stride=1, padding=3, groups=1,
                                            deformable_groups=1).cuda()
        self.dcn2 = ModulatedDeformConvPack(64, 128, kernel_size=(4, 4), stride=2, padding=1, groups=1,
                                            deformable_groups=1).cuda()
        self.dcn3 = ModulatedDeformConvPack(128, 256, kernel_size=(4, 4), stride=2, padding=1, groups=1,
                                            deformable_groups=1).cuda()
        self.IN1 = nn.InstanceNorm2d(64)
        self.IN2 = nn.InstanceNorm2d(128)
        self.IN3 = nn.InstanceNorm2d(256)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x, _ = self.dcn1(x, x)
        x = self.IN1(x)
        x = self.activation(x)
        skip1 = x

        x, _ = self.dcn2(x, x)
        x = self.IN2(x)
        x = self.activation(x)
        skip2 = x

        x, _ = self.dcn3(x, x)
        x = self.IN3(x)
        x = self.activation(x)
        x = self.model(x)
        return x, skip1, skip2


def tst_ContentEncoder():
    device = 'cuda:0'
    contentEncoder = ContentEncoder().to(device)
    content_image = torch.randn(size=(8, 3, 96, 96)).to(device)
    x, skip1, skip2 = contentEncoder(content_image)
    # [8, 256, 24, 24], [8, 64, 96, 96], [8, 128, 48, 48]
    print(x.shape, skip1.shape, skip2.shape)


if __name__ == '__main__':
    tst_ContentEncoder()

