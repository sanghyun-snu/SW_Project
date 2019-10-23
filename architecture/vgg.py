import torch.nn as nn
from torch.utils.model_zoo import load_url

__all__ = ['VGG', 'vgg16_gap']

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True, **kwargs):
        super(VGG, self).__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(1024, num_classes, kernel_size=1)

        self.feature_map = None
        self.pred = None
        if init_weights:
            self._initialize_weights()

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        self.feature_map = x

        x = self.avgpool(x)

        x = self.fc(x)
        x = x.view(x.size(0), -1)

        self.pred = x
        return x

    def get_cam(self):
        return self.feature_map, self.pred

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


def make_layers(cfg, batch_norm=False, **kwargs):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs = {
    'D_GAP': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 512, 512, 512],
}


def make_aligned_state_dict(state_dict, model_named_param):
    pretrained_keys = list()
    for key in state_dict.keys():
        if key.startswith('features.'):
            pretrained_keys.append(int(key.strip().split('.')[1].strip()))
    pretrained_keys = sorted(list(set(pretrained_keys)), reverse=True)

    model_keys = list()
    for name, param in model_named_param:
        if name.startswith('features.'):
            model_keys.append(int(name.strip().split('.')[1].strip()))
    model_keys = sorted(list(set(model_keys)), reverse=True)

    for key1, key2 in zip(pretrained_keys, model_keys):
        old_key = 'features.' + str(key1) + '.weight'
        new_key = 'features.' + str(key2) + '.weight'
        state_dict[new_key] = state_dict.pop(old_key)

        old_key = 'features.' + str(key1) + '.bias'
        new_key = 'features.' + str(key2) + '.bias'
        state_dict[new_key] = state_dict.pop(old_key)

    return state_dict


def remove_layer(state_dict, keyword):
    keys = [key for key in state_dict.keys()]
    for key in keys:
        if keyword in key:
            state_dict.pop(key)
    return state_dict


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    kwargs['init_weights'] = True
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm, **kwargs), **kwargs)
    if pretrained:
        state_dict = load_url(model_urls[arch], progress=progress)
        # remove classifier part
        state_dict = remove_layer(state_dict, 'classifier.')
        state_dict = make_aligned_state_dict(state_dict, model.named_parameters())
        model.load_state_dict(state_dict, strict=False)
    return model

def vgg16_gap(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D_GAP', False, pretrained, progress, **kwargs)

