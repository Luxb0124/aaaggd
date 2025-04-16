import os
import math
import torch
import torch.nn.init as init
from torch import nn
from torchvision.ops import DeformConv2d
import sys
sys.path.append(os.path.dirname(__file__))
from blocks import LinearBlock, Conv2dBlock, ResBlocks
from attentions import OffsetAttS2CInter, OffsetAttC2SInter


def assign_adain_params(adain_params, model):
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaIN2d":
            mean = adain_params[:, :m.num_features]
            std = adain_params[:, m.num_features:2*m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2*m.num_features:
                adain_params = adain_params[:, 2*m.num_features:]


def get_num_adain_params(model):
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaIN2d":
            num_adain_params += 2*m.num_features
    return num_adain_params


class MLP(nn.Module):
    def __init__(self, nf_in, nf_out, nf_mlp, num_blocks, norm, act, use_sn=False):
        super(MLP, self).__init__()
        self.model = nn.ModuleList()
        nf = nf_mlp
        self.model.append(LinearBlock(nf_in, nf, norm=norm, act=act, use_sn=use_sn))
        for _ in range(num_blocks - 2):
            self.model.append(LinearBlock(nf, nf, norm=norm, act=act, use_sn=use_sn))
        self.model.append(LinearBlock(nf, nf_out, norm='none', act='none', use_sn=use_sn))
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


class FASICDSCDecoder(nn.Module):
    # Feature Attention Style Inject Content Deformation Skip Connection Decoder
    def __init__(self, nf_dec=256, n_res=2, res_norm='adain', dec_norm='adain', act='relu', pad='reflect', use_sn=False):
        super(FASICDSCDecoder, self).__init__()
        print("Init FASICDSCDecoder")

        nf = nf_dec
        self.model = nn.ModuleList()
        self.model.append(ResBlocks(n_res, nf, res_norm, act, pad, use_sn=use_sn))

        self.model.append(nn.Upsample(scale_factor=2))
        self.model.append(Conv2dBlock(nf, nf // 2, 5, 1, 2, norm=dec_norm, act=act, pad_type=pad, use_sn=use_sn))
        nf //= 2

        self.model.append(nn.Upsample(scale_factor=2))
        self.model.append(Conv2dBlock(2 * nf, nf // 2, 5, 1, 2, norm=dec_norm, act=act, pad_type=pad, use_sn=use_sn))
        nf //= 2

        self.model.append(Conv2dBlock(2 * nf, 3, 7, 1, 3, norm='none', act='tanh', pad_type=pad, use_sn=use_sn))
        self.model = nn.Sequential(*self.model)

        self.sic_offset = OffsetAttS2CInter(c_in_channels=64, s_in_channels=64 * 2)
        self.sic_2_offset = OffsetAttS2CInter(c_in_channels=128, s_in_channels=128 * 2)

        self.dcn = DeformConv2d(in_channels=64 * 2, out_channels=64, kernel_size=(3, 3), stride=1, padding=1, dilation=1,)
        self.dcn_2 = DeformConv2d(in_channels=128 * 2, out_channels=128, kernel_size=(3, 3), stride=1, padding=1,
                                  dilation=1, )

    def forward(self, x, skip1, skip2):
        # [8, 256, 24, 24], [8, 64, 96, 96], [8, 128, 48, 48]
        output = x
        total_offset = 0
        for i in range(len(self.model)):
            output = self.model[i](output)

            if i == 2:
                deformable_concat = torch.cat((output, skip2), dim=1).contiguous()
                offset2 = self.sic_2_offset(content_hidden_states=skip2, style_hidden_states=deformable_concat)
                offset2 = offset2.contiguous()
                concat_pre = self.dcn_2(deformable_concat, offset2)
                output = torch.cat((concat_pre, output), dim=1)
                offset_sum = torch.mean(torch.abs(offset2))
                total_offset += offset_sum

            if i == 4:
                deformable_concat = torch.cat((output, skip1), dim=1).contiguous()
                offset1 = self.sic_offset(content_hidden_states=skip1, style_hidden_states=deformable_concat)
                offset1 = offset1.contiguous()
                concat_pre = self.dcn(deformable_concat, offset1)
                output = torch.cat((concat_pre, output), dim=1)
                offset_sum = torch.mean(torch.abs(offset1))
                total_offset += offset_sum

        total_offset = total_offset / 2
        return output, total_offset


class FACISDSCDecoder(nn.Module):
    # Feature Attention Content Inject Style Deformation Skip Connection Decoder
    def __init__(self, nf_dec=256, n_res=2, res_norm='adain', dec_norm='adain', act='relu', pad='reflect', use_sn=False):
        super(FACISDSCDecoder, self).__init__()
        print("Init FACISDSCDecoder")

        nf = nf_dec
        self.model = nn.ModuleList()
        self.model.append(ResBlocks(n_res, nf, res_norm, act, pad, use_sn=use_sn))

        self.model.append(nn.Upsample(scale_factor=2))
        self.model.append(Conv2dBlock(nf, nf // 2, 5, 1, 2, norm=dec_norm, act=act, pad_type=pad, use_sn=use_sn))
        nf //= 2

        self.model.append(nn.Upsample(scale_factor=2))
        self.model.append(Conv2dBlock(2 * nf, nf // 2, 5, 1, 2, norm=dec_norm, act=act, pad_type=pad, use_sn=use_sn))
        nf //= 2

        self.model.append(Conv2dBlock(2 * nf, 3, 7, 1, 3, norm='none', act='tanh', pad_type=pad, use_sn=use_sn))
        self.model = nn.Sequential(*self.model)

        self.sic_offset = OffsetAttC2SInter(c_in_channels=64, s_in_channels=64 * 2)
        self.sic_2_offset = OffsetAttC2SInter(c_in_channels=128, s_in_channels=128 * 2)

        self.dcn = DeformConv2d(in_channels=64 * 2, out_channels=64, kernel_size=(3, 3), stride=1, padding=1, dilation=1,)
        self.dcn_2 = DeformConv2d(in_channels=128 * 2, out_channels=128, kernel_size=(3, 3), stride=1, padding=1,
                                  dilation=1, )

    def forward(self, x, skip1, skip2):
        # [8, 256, 24, 24], [8, 64, 96, 96], [8, 128, 48, 48]
        output = x
        total_offset = 0
        for i in range(len(self.model)):
            output = self.model[i](output)

            if i == 2:
                deformable_concat = torch.cat((output, skip2), dim=1).contiguous()
                offset2 = self.sic_2_offset(content_hidden_states=skip2, style_hidden_states=deformable_concat)
                offset2 = offset2.contiguous()
                concat_pre = self.dcn_2(deformable_concat, offset2)
                output = torch.cat((concat_pre, output), dim=1)
                offset_sum = torch.mean(torch.abs(offset2))
                total_offset += offset_sum

            if i == 4:
                deformable_concat = torch.cat((output, skip1), dim=1).contiguous()
                offset1 = self.sic_offset(content_hidden_states=skip1, style_hidden_states=deformable_concat)
                offset1 = offset1.contiguous()
                concat_pre = self.dcn(deformable_concat, offset1)
                output = torch.cat((concat_pre, output), dim=1)
                offset_sum = torch.mean(torch.abs(offset1))
                total_offset += offset_sum

        total_offset = total_offset / 2
        return output, total_offset


def tst_FASICDSCDecoder():
    print('tst_FASICDSCDecoder')
    device = 'cuda:0'
    x = torch.randn(size=(8, 256, 24, 24)).to(device)
    sty = torch.randn(size=(8, 128)).to(device)
    skip1 = torch.randn(size=(8, 64, 96, 96)).to(device)
    skip2 = torch.randn(size=(8, 128, 48, 48)).to(device)

    decoder = FASICDSCDecoder().to(device)
    # print(decoder)

    mlp = MLP(128, get_num_adain_params(decoder), 256, 3, 'none', 'relu').to(device)
    # print(mlp)

    adapt_params = mlp(sty)
    assign_adain_params(adapt_params, decoder)
    output, total_offset = decoder(x, skip1, skip2)
    print(output.shape, total_offset)


def tst_FACISDSCDecoder():
    print('tst_FACISDSCDecoder')
    device = 'cuda:0'
    x = torch.randn(size=(8, 256, 24, 24)).to(device)
    sty = torch.randn(size=(8, 128)).to(device)
    skip1 = torch.randn(size=(8, 64, 96, 96)).to(device)
    skip2 = torch.randn(size=(8, 128, 48, 48)).to(device)

    decoder = FACISDSCDecoder().to(device)
    # print(decoder)

    mlp = MLP(128, get_num_adain_params(decoder), 256, 3, 'none', 'relu').to(device)
    # print(mlp)

    adapt_params = mlp(sty)
    assign_adain_params(adapt_params, decoder)
    output, total_offset = decoder(x, skip1, skip2)
    print(output.shape, total_offset)


if __name__ == '__main__':
    tst_FASICDSCDecoder()
    tst_FACISDSCDecoder()
