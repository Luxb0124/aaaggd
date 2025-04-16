import os
import torch
from torch import nn
import torch.nn.functional as F


cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'vgg19cut': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'N'],
}


class GuidingNet(nn.Module):
    def __init__(self, sty_dim, output_k, load_path=None, load_key='C_EMA_state_dict'):
        super(GuidingNet, self).__init__()
        # network layers setting
        self.features = make_layers(cfg['vgg11'], True)

        self.disc = nn.Linear(512, output_k)
        self.cont = nn.Linear(512, sty_dim)

        if load_path is None:
            self._initialize_weights()
        else:
            self.load_model(load_path, load_key)

    def load_model(self, load_path='default', load_key='C_EMA_state_dict'):
        if load_path == 'default':
            load_path = os.path.join(os.path.dirname(__file__), 'model_250.ckpt')
        assert os.path.exists(load_path), f'{load_path} is not exists...'
        checkpoint = torch.load(load_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint[load_key]
        self.load_state_dict(state_dict=state_dict)

    def forward(self, x, sty=False):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        flat = x.view(x.size(0), -1)
        cont = self.cont(flat)
        if sty:
            return cont
        disc = self.disc(flat)
        return {'cont': cont, 'disc': disc}

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def moco(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        flat = x.view(x.size(0), -1)
        cont = self.cont(flat)
        return cont

    def iic(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        flat = x.view(x.size(0), -1)
        disc = self.disc(flat)
        return disc


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
            else:
                layers += [conv2d, nn.ReLU(inplace=False)]
            in_channels = v
    return nn.Sequential(*layers)


if __name__ == '__main__':
    import torch
    C = GuidingNet(64)
    x_in = torch.randn(4, 3, 64, 64)
    sty = C.moco(x_in)
    cls = C.iic(x_in)
    print(sty.shape, cls.shape)
