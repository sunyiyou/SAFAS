import torch.nn as nn
import math
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
__all__ = ['ResNet', 'resnet18', 'resnet50', ]


normalization = nn.BatchNorm2d

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = normalization(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = normalization(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out

    def forward_masked(self, x, mask=None):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)


        out = out + residual
        if mask is not None:
            out = out * mask[None,:,None,None]# + self.bn2.bias[None,:,None,None] * (1 - mask[None,:,None,None])

        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = normalization(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = normalization(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = normalization(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class AbstractResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, maxiter=-1):
        super(AbstractResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = normalization(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def _initial_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                normalization(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        # x = self.layer1(x)
        return x

    def forward(self, x, ):
        x = self.features(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def load_state_dict(self, state_dict, strict=True):
        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(self)

        if strict:
            error_msg = ''
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0, 'Unexpected key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in unexpected_keys)))
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0, 'Missing key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in missing_keys)))

        if len(error_msgs) > 0:
            print('Warning(s) in loading state_dict for {}:\n\t{}'.format(self.__class__.__name__, "\n\t".join(error_msgs)))



class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim=1, bn=False):
        super(LogisticRegression, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1, input_dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.bias.data.zero_()
        self.reset_parameters()
        # self.register_parameter('bias', None)
        self.bn = bn
        if bn:
            self.bn_layer = nn.BatchNorm1d(1)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, scale=1.0):
        logit = F.linear(x, self.weight, self.bias) * scale
        if self.bn:
            logit = self.bn_layer(logit)
        outputs = torch.sigmoid(logit)
        return outputs

class NormedLogisticRegression(nn.Module):

    def __init__(self, in_features, out_features, bn=False, use_bias=True):
        super(NormedLogisticRegression, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1, in_features))
        # self.weight.data.uniform_(0, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.bias = nn.Parameter(torch.Tensor(out_features))

        self.use_bias = use_bias
        self.bn = bn
        if bn:
            self.bn_layer = nn.BatchNorm1d(out_features)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias.data.zero_()


    def forward(self, x, scale=1.0):
        if self.use_bias:
            out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0).t()) * scale + self.bias[:, None]
        else:
            out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0).t()) * scale

        if self.bn:
            out = self.bn_layer(out)

        outputs = torch.sigmoid(out)
        return outputs

normalizer = lambda x: x / (torch.norm(x, dim=-1, keepdim=True) + 1e-10)

class ResNet(AbstractResNet):

    def __init__(self, block, layers, num_classes=2, max_iter=-1, normalized=True, use_bias=False, simsiam=False):
        super(ResNet, self).__init__(block, layers, num_classes)
        feat_dim = 512

        if normalized:
            self.fc0 = NormedLogisticRegression(feat_dim * block.expansion, 1, use_bias=use_bias)
            self.fc1 = NormedLogisticRegression(feat_dim * block.expansion, 1, use_bias=use_bias)
            self.fc2 = NormedLogisticRegression(feat_dim * block.expansion, 1, use_bias=use_bias)

        else:
            self.fc0 = LogisticRegression(feat_dim * block.expansion, 1)
            self.fc1 = LogisticRegression(feat_dim * block.expansion, 1)
            self.fc2 = LogisticRegression(feat_dim * block.expansion, 1)

        self.bn_scale = nn.BatchNorm1d(1)
        self.fc_scale = nn.Linear(feat_dim * block.expansion, 1)
        self.proj = nn.Linear(feat_dim * block.expansion, feat_dim * block.expansion)

        self._initial_weight()

        if simsiam:
            self.projection = nn.Sequential(
                                            nn.Linear(feat_dim, feat_dim, bias=False),
                                            nn.BatchNorm1d(feat_dim),
                                            nn.ReLU(inplace=True), # first layer
                                            nn.Linear(feat_dim, feat_dim, bias=False),
                                            nn.BatchNorm1d(feat_dim),
                                            nn.ReLU(inplace=True), # second layer
                                            nn.Linear(feat_dim, feat_dim, bias=False),
                                            nn.BatchNorm1d(feat_dim, affine=False)
                                            ) # output layer

            proj_feat_dim = 128
            self.predictor = nn.Sequential(nn.Linear(feat_dim, proj_feat_dim, bias=False),
                                            nn.BatchNorm1d(proj_feat_dim),
                                            nn.ReLU(inplace=True), # hidden layer
                                            nn.Linear(proj_feat_dim, feat_dim)) # output layer
    def simsiam_loss(self, feat):
        proj_feat = self.projection(feat)
        b = len(proj_feat)
        z1 = proj_feat[:b//2]
        z2 = proj_feat[b//2:]
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        z1 = z1.detach()
        z2 = z2.detach()
        loss = -(F.cosine_similarity(p1, z2).mean() + F.cosine_similarity(p2, z1).mean()) * 0.5
        return loss

    def snapshot_weight(self):
        self.w0_old = self.fc0.weight.data.clone()
        self.w1_old = self.fc1.weight.data.clone()
        self.w2_old = self.fc2.weight.data.clone()

    def update_weight_d2(self, alpha=0.99):
        beta0 = normalizer(self.fc0.weight.data.squeeze())
        beta1 = normalizer(self.fc1.weight.data.squeeze())

        self.fc0.weight.data = (normalizer(alpha * beta0 + (1 - alpha) * beta1) * torch.norm(self.fc0.weight.data)).unsqueeze(0)
        self.fc1.weight.data = (normalizer(alpha * beta1 + (1 - alpha) * beta0) * torch.norm(self.fc1.weight.data)).unsqueeze(0)

        return (beta0 * beta1).sum()

    def update_weight_v4(self, alpha=0.99):
        beta0 = normalizer(self.fc0.weight.data.squeeze())
        beta1 = normalizer(self.fc1.weight.data.squeeze())
        beta2 = normalizer(self.fc2.weight.data.squeeze())

        beta0_fr =  min([((beta0 * beta1).sum().item(), 0, beta1), ((beta0 * beta2).sum().item(), 1, beta2)])[2]
        beta1_fr =  min([((beta1 * beta0).sum().item(), 0, beta0), ((beta1 * beta2).sum().item(), 1, beta2)])[2]
        beta2_fr =  min([((beta2 * beta0).sum().item(), 0, beta0), ((beta2 * beta1).sum().item(), 1, beta1)])[2]

        self.fc0.weight.data = (normalizer(alpha * beta0 + (1 - alpha) * beta0_fr) * torch.norm(self.fc0.weight.data)).unsqueeze(0)
        self.fc1.weight.data = (normalizer(alpha * beta1 + (1 - alpha) * beta1_fr) * torch.norm(self.fc1.weight.data)).unsqueeze(0)
        self.fc2.weight.data = (normalizer(alpha * beta2 + (1 - alpha) * beta2_fr) * torch.norm(self.fc2.weight.data)).unsqueeze(0)

        return ((beta0 * beta1).sum() + (beta2 * beta1).sum() + (beta0 * beta2).sum()) / 3

    def forward(self, x, fc_params=None, out_type='ce_sep', scale=None):
        feat = self.features(x)
        # b, c, w, h = feat.shape
        # feat = torch.matmul(feat.view(b, c, w*h), torch.transpose(feat.view(b, c, w*h), 2, 1)).view(b, c * c) / w*h

        feat = self.avgpool(feat)
        feat = feat.view(feat.size(0), -1)
        # feat = self.proj(feat)
        if scale is None:
            scale = torch.exp(self.bn_scale(self.fc_scale(feat)))
        else:
            scale = torch.ones_like(torch.exp(self.bn_scale(self.fc_scale(feat)))) * scale


        if out_type == 'ce_sep':
            return [self.fc0(feat, scale), self.fc1(feat, scale), self.fc2(feat, scale)]
        if out_type == 'ce':
            return feat, self.fc0(feat, scale)
        if out_type == 'feat':
            return feat, scale
        if out_type == 'all':
            return feat, feat, (self.fc0(feat, scale) + self.fc1(feat, scale)+ self.fc2(feat, scale)) / 3
        raise RuntimeError("None supported out_type")


def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, x):
        return F.normalize(x, dim=-1)

