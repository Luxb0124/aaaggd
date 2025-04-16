import os
import torch

from torch import nn
import sys

sys.path.append(os.path.dirname(__file__))
from encoders import ContentEncoder, weights_init
from decoders import FACISDSCDecoder, FASICDSCDecoder, MLP, assign_adain_params, get_num_adain_params


class BaseGenerator(nn.Module):
    def __init__(self, cnt_encoder=None, decoder=None, sty_dim=128, load_path=None, load_key='G_EMA_state_dict'):
        super(BaseGenerator, self).__init__()
        print("Init BaseGenerator")

        self.nf_mlp = 256

        self.adaptive_param_getter = get_num_adain_params
        self.adaptive_param_assign = assign_adain_params

        self.cnt_encoder = cnt_encoder
        self.decoder = decoder
        self.mlp = MLP(sty_dim, self.adaptive_param_getter(self.decoder), self.nf_mlp, 3, 'none', 'relu')

        if load_path is None:
            self.apply(weights_init('kaiming'))
        else:
            self.load_model(load_path, load_key)

    def load_model(self, load_path='default', load_key='G_EMA_state_dict'):
        if load_path == 'default':
            load_path = os.path.join(os.path.dirname(__file__), 'model_250.ckpt')
        assert os.path.exists(load_path), f'{load_path} is not exists...'
        checkpoint = torch.load(load_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint[load_key]
        self.load_state_dict(state_dict=state_dict)

    def forward(self, x_src, s_ref):
        c_src, skip1, skip2 = self.cnt_encoder(x_src)
        x_out = self.decode(c_src, s_ref, skip1, skip2)
        return x_out

    def decode(self, cnt, sty, skip1, skip2):
        adapt_params = self.mlp(sty)
        self.adaptive_param_assign(adapt_params, self.decoder)
        out = self.decoder(cnt, skip1, skip2)
        return out

    def _initialize_weights(self, mode='fan_in'):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()


class SICGenerator(BaseGenerator):
    def __init__(self, nf_enc=64, n_downs=2, n_res=2, enc_norm='in', act='relu', pad='reflect',
                 nf_dec=256, sty_norm='adain', dec_norm='adain', use_sn=False, sty_dim=128,
                 load_path=None, load_key='G_EMA_state_dict'):
        print("Init SICGenerator")
        cnt_encoder = ContentEncoder(nf_cnt=nf_enc, n_downs=n_downs, n_res=n_res, norm=enc_norm, act=act, pad=pad,
                                     use_sn=use_sn)
        decoder = FASICDSCDecoder(nf_dec=nf_dec, n_res=n_res, res_norm=sty_norm, dec_norm=dec_norm, act=act, pad=pad,
                                  use_sn=use_sn)
        super(SICGenerator, self).__init__(cnt_encoder=cnt_encoder, decoder=decoder, sty_dim=sty_dim,
                                           load_path=load_path, load_key=load_key)


class CISGenerator(BaseGenerator):
    def __init__(self, nf_enc=64, n_downs=2, n_res=2, enc_norm='in', act='relu', pad='reflect',
                 nf_dec=256, sty_norm='adain', dec_norm='adain', use_sn=False, sty_dim=128,
                 load_path=None, load_key='G_EMA_state_dict'):
        print("Init CISGenerator")
        cnt_encoder = ContentEncoder(nf_cnt=nf_enc, n_downs=n_downs, n_res=n_res, norm=enc_norm, act=act, pad=pad,
                                     use_sn=use_sn)
        decoder = FACISDSCDecoder(nf_dec=nf_dec, n_res=n_res, res_norm=sty_norm, dec_norm=dec_norm, act=act, pad=pad,
                                  use_sn=use_sn)
        super(CISGenerator, self).__init__(cnt_encoder=cnt_encoder, decoder=decoder, sty_dim=sty_dim,
                                           load_path=load_path, load_key=load_key)


def tst_SICGenerator():
    print('tst_SICGenerator')
    device = 'cuda:0'
    cnt_encoder = ContentEncoder().to(device)
    decoder = FASICDSCDecoder().to(device)
    generator = BaseGenerator(cnt_encoder=cnt_encoder, decoder=decoder).to(device)
    content_img = torch.randn(size=(8, 3, 96, 96)).to(device)
    feature_sty_ref = torch.randn(size=(8, 128)).to(device)
    feature_cnt_src, feature_cnt_skip1, feature_cnt_skip2 = generator.cnt_encoder(content_img)
    generated_img, _ = generator.decode(feature_cnt_src, feature_sty_ref, feature_cnt_skip1, feature_cnt_skip2)
    print(generated_img.shape)


def tst_CISGenerator():
    print('tst_CISGenerator')
    device = 'cuda:0'
    cnt_encoder = ContentEncoder().to(device)
    decoder = FACISDSCDecoder().to(device)
    generator = BaseGenerator(cnt_encoder=cnt_encoder, decoder=decoder).to(device)
    content_img = torch.randn(size=(8, 3, 96, 96)).to(device)
    feature_sty_ref = torch.randn(size=(8, 128)).to(device)
    feature_cnt_src, feature_cnt_skip1, feature_cnt_skip2 = generator.cnt_encoder(content_img)
    generated_img, _ = generator.decode(feature_cnt_src, feature_sty_ref, feature_cnt_skip1, feature_cnt_skip2)
    print(generated_img.shape)


if __name__ == '__main__':
    tst_SICGenerator()
    tst_CISGenerator()
