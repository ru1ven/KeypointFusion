import torch.nn as nn
import math
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor

from model.fusion_layer import RGBDFusion

BN_MOMENTUM = 0.1


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c0 = self.maxpool(x)

        c1 = self.layer1(c0)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        return c0, c1, c2, c3, c4


import functools
import time
import torch
import torch.nn as nn
import config




class RGBD_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_layer=None,
                 bn_eps=1e-5, bn_momentum=0.1, downsample=None, inplace=True):
        super(RGBD_BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)

        self.depth_conv1 = conv3x3(inplanes, planes, stride)
        self.depth_bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.depth_relu = nn.ReLU(inplace=inplace)
        self.depth_relu_inplace = nn.ReLU(inplace=True)
        self.depth_conv2 = conv3x3(planes, planes)
        self.depth_bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)

        self.downsample = downsample
        self.depth_downsample = downsample

        self.stride = stride
        self.inplace = inplace

    def forward(self, x):
        #first path
        x1 = x[0]

        residual1 = x1

        out1 = self.conv1(x1)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        out1 = self.conv2(out1)
        out1 = self.bn2(out1)

        if self.downsample is not None:
            residual1 = self.downsample(x1)

        #second path
        x2 = x[1]
        residual2 = x2

        out2 = self.depth_conv1(x2)
        out2 = self.depth_bn1(out2)
        out2 = self.depth_relu(out2)

        out2 = self.depth_conv2(out2)
        out2 = self.depth_bn2(out2)

        if self.depth_downsample is not None:
            residual2 = self.depth_downsample(x2)

        out1 += residual1
        out2 += residual2

        out1 = self.relu_inplace(out1)
        out2 = self.relu_inplace(out2)

        return [out1, out2]


class RGBD_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,
                 norm_layer=None, bn_eps=1e-5, bn_momentum=0.1,
                 downsample=None, inplace=True):
        super(RGBD_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = norm_layer(planes * self.expansion, eps=bn_eps,
                              momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample


        self.depth_conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.depth_bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.depth_conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.depth_bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.depth_conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.depth_bn3 = norm_layer(planes * self.expansion, eps=bn_eps,
                              momentum=bn_momentum)
        self.depth_relu = nn.ReLU(inplace=inplace)
        self.depth_relu_inplace = nn.ReLU(inplace=True)
        self.depth_downsample = downsample


        self.stride = stride
        self.inplace = inplace

    def forward(self, x):
        # first path
        x1 = x[0]
        residual1 = x1

        out1 = self.conv1(x1)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        out1 = self.conv2(out1)
        out1 = self.bn2(out1)
        out1 = self.relu(out1)

        out1 = self.conv3(out1)
        out1 = self.bn3(out1)

        if self.downsample is not None:
            residual1 = self.downsample(x1)

        # second path
        x2 = x[1]
        residual2 = x2

        out2 = self.depth_conv1(x2)
        out2 = self.depth_bn1(out2)
        out2 = self.depth_relu(out2)

        out2 = self.depth_conv2(out2)
        out2 = self.depth_bn2(out2)
        out2 = self.depth_relu(out2)

        out2 = self.depth_conv3(out2)
        out2 = self.depth_bn3(out2)

        if self.depth_downsample is not None:
            residual2 = self.depth_downsample(x2)

        out1 += residual1
        out2 += residual2
        out1 = self.relu_inplace(out1)
        out2 = self.relu_inplace(out2)

        return [out1, out2]



class RGBD_ResNet(nn.Module):

    def __init__(self, block, layers, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 bn_momentum=0.1, inplace=True):
        self.inplanes = 64
        super(RGBD_ResNet, self).__init__()


        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.depth_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.bn1 = norm_layer(64, eps=bn_eps, momentum=bn_momentum)
        self.depth_bn1 = norm_layer( 64, eps=bn_eps, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.depth_relu = nn.ReLU(inplace=inplace)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.depth_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, norm_layer, 64, layers[0],
                                       inplace,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer2 = self._make_layer(block, norm_layer, 128, layers[1],
                                       inplace, stride=2,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer3 = self._make_layer(block, norm_layer, 256, layers[2],
                                       inplace, stride=2,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer4 = self._make_layer(block, norm_layer, 512, layers[3],
                                       inplace, stride=2,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)

        # fusion_layer
        self.RGBDfusion_layers = nn.ModuleList([
            RGBDFusion(in_planes=64, out_planes=64, bn_momentum=0.1),
            RGBDFusion(in_planes=128, out_planes=128, bn_momentum=0.1),
            RGBDFusion(in_planes=256, out_planes=256, bn_momentum=0.1),
            RGBDFusion(in_planes=512, out_planes=512, bn_momentum=0.1)
        ])

    def _make_layer(self, block, norm_layer, planes, blocks, inplace=True,
                    stride=1, bn_eps=1e-5, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion, eps=bn_eps,
                           momentum=bn_momentum),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, norm_layer, bn_eps,
                            bn_momentum, downsample, inplace))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                norm_layer=norm_layer, bn_eps=bn_eps,
                                bn_momentum=bn_momentum, inplace=inplace))

        return nn.Sequential(*layers)

    def forward(self, x1, x2,writer=None,ii=0):
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)


        x2 = self.depth_conv1(x2)
        x2 = self.depth_bn1(x2)
        x2 = self.depth_relu(x2)
        x2 = self.depth_maxpool(x2)

        x = [x1, x2]
        blocks = []
        merges = []
        x = self.layer1(x)
        x, merge = self.RGBDfusion_layers[0](x,writer,ii,0)
        blocks.append(x)
        merges.append(merge)

        x = self.layer2(x)
        x, merge = self.RGBDfusion_layers[1](x,writer,ii,1)
        blocks.append(x)
        merges.append(merge)

        x = self.layer3(x)
        x, merge = self.RGBDfusion_layers[2](x,writer,ii,2)
        blocks.append(x)
        merges.append(merge)

        x = self.layer4(x)
        x, merge = self.RGBDfusion_layers[3](x,writer,ii,3)
        blocks.append(x)
        merges.append(merge)

        return blocks, merges



def load_dualpath_model(model, model_file, is_restore=False):
    # load raw state_dict
    t_start = time.time()
    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file)


        if 'model' in raw_state_dict.keys():
            raw_state_dict = raw_state_dict['model']
    else:
        raw_state_dict = model_file
    # copy to  depth backbone
    state_dict = {}
    for k, v in raw_state_dict.items():
        state_dict[k.replace('.bn.', '.')] = v
        if k.find('conv1') >= 0:
            state_dict[k] = v
            state_dict[k.replace('conv1', 'depth_conv1')] = v
        if k.find('conv2') >= 0:
            state_dict[k] = v
            state_dict[k.replace('conv2', 'depth_conv2')] = v
        if k.find('conv3') >= 0:
            state_dict[k] = v
            state_dict[k.replace('conv3', 'depth_conv3')] = v
        if k.find('bn1') >= 0:
            state_dict[k] = v
            state_dict[k.replace('bn1', 'depth_bn1')] = v
        if k.find('bn2') >= 0:
            state_dict[k] = v
            state_dict[k.replace('bn2', 'depth_bn2')] = v
        if k.find('bn3') >= 0:
            state_dict[k] = v
            state_dict[k.replace('bn3', 'depth_bn3')] = v
        if k.find('downsample') >= 0:
            state_dict[k] = v
            state_dict[k.replace('downsample', 'depth_downsample')] = v
    t_ioend = time.time()

    if is_restore:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False)
    # ckpt_keys = set(state_dict.keys())
    # own_keys = set(model.state_dict().keys())
    # missing_keys = own_keys - ckpt_keys
    # unexpected_keys = ckpt_keys - own_keys
    #
    # if len(missing_keys) > 0:
    #     logger.warning('Missing key(s) in state_dict: {}'.format(
    #         ', '.join('{}'.format(k) for k in missing_keys)))
    #
    # if len(unexpected_keys) > 0:
    #     logger.warning('Unexpected key(s) in state_dict: {}'.format(
    #         ', '.join('{}'.format(k) for k in unexpected_keys)))

    del state_dict
    t_end = time.time()
    # logger.info(
    #     "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
    #         t_ioend - t_start, t_end - t_ioend))

    return model

def resnet18(pretrained_model=None, **kwargs):
    model = RGBD_ResNet(RGBD_BasicBlock, [2, 2, 2, 2], **kwargs)

    if pretrained_model is not None:
        model = load_dualpath_model(model, pretrained_model)
    return model


def resnet34(pretrained_model=None, **kwargs):
    model = RGBD_ResNet(RGBD_BasicBlock, [3, 4, 6, 3], **kwargs)

    if pretrained_model is not None:
        model = load_dualpath_model(model, pretrained_model)
    return model


def resnet50(pretrained_model=None, **kwargs):
    model = RGBD_ResNet(RGBD_Bottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained_model is not None:
        model = load_dualpath_model(model, pretrained_model)
    return model


def resnet101(pretrained_model=None, **kwargs):
    model = RGBD_ResNet(RGBD_Bottleneck, [3, 4, 23, 3], **kwargs)

    if pretrained_model is not None:
        model = load_dualpath_model(model, pretrained_model)
    return model


def resnet152(pretrained_model=None, **kwargs):
    model = RGBD_ResNet(RGBD_Bottleneck, [3, 8, 36, 3], **kwargs)

    if pretrained_model is not None:
        model = load_dualpath_model(model, pretrained_model)
    return model