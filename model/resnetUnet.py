import math
import torch
import torch.nn as nn
from model.resnet import BasicBlock, Bottleneck, ResNet, RGBD_BasicBlock, RGBD_Bottleneck, RGBD_ResNet, \
    load_dualpath_model
from model.hourglass import Residual
import torchvision
from model.fusion_layer import RGBDFusion

BN_MOMENTUM = 0.1

resnet = {18: (BasicBlock, [2, 2, 2, 2]),
          50: (Bottleneck, [3, 4, 6, 3]),
          101: (Bottleneck, [3, 4, 23, 3]),
          152: (Bottleneck, [3, 8, 36, 3])
          }

resnet_rgbd = {18: (RGBD_BasicBlock, [2, 2, 2, 2]),
          50: (RGBD_Bottleneck, [3, 4, 6, 3]),
          101: (RGBD_Bottleneck, [3, 4, 23, 3]),
          152: (RGBD_Bottleneck, [3, 8, 36, 3])
          }


def conv_bn_relu(in_planes, out_planes, kernel):
    return nn.Sequential(
        nn.Conv2d(
                in_channels=in_planes,
                out_channels=out_planes,
                kernel_size=kernel,
                stride=1,
                padding=1,
                bias=False),
        nn.BatchNorm2d(out_planes, momentum=0.1),
        nn.ReLU(inplace=True)
    )


def convtranspose_bn_relu(in_planes, out_planes, kernel):
    return nn.Sequential(
        nn.ConvTranspose2d(
                in_channels=in_planes,
                out_channels=out_planes,
                kernel_size=kernel,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False),
        nn.BatchNorm2d(out_planes, momentum=0.1),
        nn.ReLU(inplace=True)
    )


class ResNetUnetAdd(nn.Module):
    def __init__(self, backbone, joint_num, deconv_dim=128, out_dim_list=[3*21, 21, 21]):
        super(ResNetUnetAdd, self).__init__()
        self.joint_num = joint_num
        self.feature_dim = [self.joint_num * 3, self.joint_num]
        layers_num = int(backbone.split('_')[-1])
        block, layers = resnet[layers_num]
        self.backbone = ResNet(block, layers)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.skip_layer4 = Residual(256*block.expansion, 512*block.expansion)
        self.up4 = nn.Sequential(Residual(512*block.expansion, 512*block.expansion),
                                 nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer4 = Residual((512+0)*block.expansion, 256*block.expansion)

        self.skip_layer3 = Residual(128*block.expansion, 256*block.expansion)
        self.up3 = nn.Sequential(Residual(256*block.expansion, 256*block.expansion),
                                  nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer3 = Residual((256+0)*block.expansion, 128*block.expansion)

        self.skip_layer2 = Residual(64*block.expansion, 128*block.expansion)
        self.up2 = nn.Sequential(Residual(128*block.expansion, 128*block.expansion),
                                  nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer2 = Residual((128+0)*block.expansion, deconv_dim)

        self.finals = nn.ModuleList()
        for out_dim in out_dim_list:
            self.finals.append(nn.Conv2d(in_channels=deconv_dim, out_channels=out_dim, kernel_size=1, stride=1))

        self.init_weights()

    def init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)

        for m in self.finals.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, img):
        device = img.device
        c0, c1, c2, c3, c4 = self.backbone(img)

        c4_up = self.up4(c4)
        c3_skip = self.skip_layer4(c3)
        c3_fusion = self.fusion_layer4(c4_up + c3_skip)
        # c3_fusion = self.fusion_layer4(torch.cat((c4_up, c3_skip), dim=1))

        c3_up = self.up3(c3_fusion)
        c2_skip = self.skip_layer3(c2)
        c2_fusion = self.fusion_layer3(c3_up + c2_skip)
        # c2_fusion = self.fusion_layer3(torch.cat((c3_up, c2_skip), dim=1))

        c2_up = self.up2(c2_fusion)
        c1_skip = self.skip_layer2(c1)
        # img_feature = self.fusion_layer2(torch.cat((c2_up, c1_skip), dim=1))
        img_feature = self.fusion_layer2(c2_up + c1_skip)

        img_result = torch.Tensor().to(device)
        for layer in self.finals:
            temp = layer(img_feature)
            img_result = torch.cat((img_result, temp), dim=1)

        return img_result, img_feature


class ResNetUnet(nn.Module):
    def __init__(self, backbone, joint_num, deconv_dim=128, out_dim_list=[3*21, 21, 21]):
        super(ResNetUnet, self).__init__()
        self.joint_num = joint_num
        self.feature_dim = [self.joint_num * 3, self.joint_num]
        layers_num = int(backbone.split('_')[-1])
        block, layers = resnet[layers_num]
        self.pre = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2, bias=False),
            # nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.inplanes = 64
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.skip_layer4 = Residual(256*block.expansion, 256*block.expansion)
        self.up4 = nn.Sequential(Residual(512*block.expansion, 512*block.expansion),
                                 nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer4 = Residual((512+256)*block.expansion, 256*block.expansion)

        self.skip_layer3 = Residual(128*block.expansion, 128*block.expansion)
        self.up3 = nn.Sequential(Residual(256*block.expansion, 256*block.expansion),
                                  nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer3 = Residual((256+128)*block.expansion, 128*block.expansion)

        self.skip_layer2 = Residual(64*block.expansion, 64*block.expansion)
        self.up2 = nn.Sequential(Residual(128*block.expansion, 128*block.expansion),
                                  nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer2 = Residual((128+64)*block.expansion, deconv_dim)

        self.finals = nn.ModuleList()
        for out_dim in out_dim_list:
            self.finals.append(nn.Conv2d(in_channels=deconv_dim, out_channels=out_dim, kernel_size=1, stride=1))

        self.init_weights()

    def init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)

        for m in self.finals.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, img):
        device = img.device
        c0 = self.pre(img)
        c1 = self.layer1(c0)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        c4_up = self.up4(c4)
        c3_skip = self.skip_layer4(c3)
        c3_fusion = self.fusion_layer4(torch.cat((c4_up, c3_skip), dim=1))

        c3_up = self.up3(c3_fusion)
        c2_skip = self.skip_layer3(c2)
        c2_fusion = self.fusion_layer3(torch.cat((c3_up, c2_skip), dim=1))

        c2_up = self.up2(c2_fusion)
        c1_skip = self.skip_layer2(c1)
        img_feature = self.fusion_layer2(torch.cat((c2_up, c1_skip), dim=1))

        img_result = torch.Tensor().to(device)
        #print(img_feature.shape)
        for layer in self.finals:
            temp = layer(img_feature)
            img_result = torch.cat((img_result, temp), dim=1)

        return img_result, img_feature, c4


class OfficialResNetUnet(nn.Module):
    def __init__(self, backbone, joint_num, pretrain=True, deconv_dim=128, out_dim_list=[3*21, 21, 21]):
        super(OfficialResNetUnet, self).__init__()
        self.joint_num = joint_num
        self.feature_dim = [self.joint_num * 3, self.joint_num]
        layers_num = int(backbone.split('-')[-1])
        block, layers = resnet[layers_num]
        self.backbone = ResNet(block, layers)
        self.skip_layer4 = Residual(256*block.expansion, 256)
        self.up4 = nn.Sequential(Residual(512*block.expansion, 512),
                                 nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer4 = Residual((512+256), 256)

        self.skip_layer3 = Residual(128*block.expansion, 128)
        self.up3 = nn.Sequential(Residual(256, 256),
                                  nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer3 = Residual((256+128), 128)

        self.skip_layer2 = Residual(64*block.expansion, 64)
        self.up2 = nn.Sequential(Residual(128, 128),
                                  nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer2 = Residual((128+64), deconv_dim)


        self.finals = nn.ModuleList()
        for out_dim in out_dim_list:
            self.finals.append(nn.Conv2d(in_channels=deconv_dim, out_channels=out_dim, kernel_size=1, stride=1))

        self.init_weights()
        if pretrain:
            if layers_num == 18:
                print('load weight from resnet-18')
                pretrain_weight = torchvision.models.resnet18(pretrained=True)
            elif layers_num == 50:
                print('load weight from resnet-50')
                pretrain_weight = torchvision.models.resnet50(pretrained=True)
            elif layers_num == 101:
                print('load weight from resnet-101')
                pretrain_weight = torchvision.models.resnet101(pretrained=True)
            self.backbone.load_state_dict(pretrain_weight.state_dict(), strict=False)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)

        for m in self.finals.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, img):
        device = img.device
        c0, c1, c2, c3, c4 = self.backbone(img)

        c4_up = self.up4(c4)
        c3_skip = self.skip_layer4(c3)
        c3_fusion = self.fusion_layer4(torch.cat((c4_up, c3_skip), dim=1))

        c3_up = self.up3(c3_fusion)
        c2_skip = self.skip_layer3(c2)
        c2_fusion = self.fusion_layer3(torch.cat((c3_up, c2_skip), dim=1))

        c2_up = self.up2(c2_fusion)
        c1_skip = self.skip_layer2(c1)
        img_feature = self.fusion_layer2(torch.cat((c2_up, c1_skip), dim=1))

        img_result = torch.Tensor().to(device)
        for layer in self.finals:
            temp = layer(img_feature)
            img_result = torch.cat((img_result, temp), dim=1)

        return img_result, img_feature


class OfficialResNetUnet_RGB2offset_3D(nn.Module):
    def __init__(self, backbone, joint_num, pretrain=True, deconv_dim=128, out_dim_list=[3*21, 21, 21]):
        super(OfficialResNetUnet_RGB2offset_3D, self).__init__()
        self.joint_num = joint_num
        self.feature_dim = [self.joint_num * 3, self.joint_num]
        layers_num = int(backbone.split('-')[-1])
        block, layers = resnet[layers_num]
        self.backbone = ResNet(block, layers)
        self.skip_layer4 = Residual(256*block.expansion, 256)
        self.up4 = nn.Sequential(Residual(512*block.expansion, 512),
                                 nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer4 = Residual((512+256), 256)

        self.skip_layer3 = Residual(128*block.expansion, 128)
        self.up3 = nn.Sequential(Residual(256, 256),
                                  nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer3 = Residual((256+128), 128)

        self.skip_layer2 = Residual(64*block.expansion, 64)
        self.up2 = nn.Sequential(Residual(128, 128),
                                  nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer2 = Residual((128+64), deconv_dim)


        self.finals = nn.ModuleList()
        for out_dim in out_dim_list:
            self.finals.append(nn.Conv2d(in_channels=deconv_dim, out_channels=out_dim, kernel_size=1, stride=1))

        self.init_weights()
        if pretrain:
            if layers_num == 18:
                print('load weight from resnet-18')
                pretrain_weight = torchvision.models.resnet18(pretrained=True)
            elif layers_num == 50:
                print('load weight from resnet-50')
                pretrain_weight = torchvision.models.resnet50(pretrained=True)
            elif layers_num == 101:
                print('load weight from resnet-101')
                pretrain_weight = torchvision.models.resnet101(pretrained=True)
            self.backbone.load_state_dict(pretrain_weight.state_dict(), strict=False)
        #self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)

        for m in self.finals.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, img):
        device = img.device
        c0, c1, c2, c3, c4 = self.backbone(img)

        c4_up = self.up4(c4)
        c3_skip = self.skip_layer4(c3)
        c3_fusion = self.fusion_layer4(torch.cat((c4_up, c3_skip), dim=1))

        c3_up = self.up3(c3_fusion)
        c2_skip = self.skip_layer3(c2)
        c2_fusion = self.fusion_layer3(torch.cat((c3_up, c2_skip), dim=1))

        c2_up = self.up2(c2_fusion)
        c1_skip = self.skip_layer2(c1)
        img_feature = self.fusion_layer2(torch.cat((c2_up, c1_skip), dim=1))

        img_result = torch.Tensor().to(device)
        for layer in self.finals:
            temp = layer(img_feature)
            img_result = torch.cat((img_result, temp), dim=1)

        return img_result, img_feature

import torch.nn.functional as F
class SoftHeatmap(nn.Module):
    def __init__(self, size, kp_num):
        super(SoftHeatmap, self).__init__()
        self.size = size
        self.beta = nn.Conv2d(kp_num, kp_num, 1, 1, 0, groups=kp_num, bias=False)
        self.wx = torch.arange(0.0, 1.0 * self.size, 1).view([1, self.size]).repeat([self.size, 1])
        self.wy = torch.arange(0.0, 1.0 * self.size, 1).view([self.size, 1]).repeat([1, self.size])
        self.wx = nn.Parameter(self.wx, requires_grad=False)
        self.wy = nn.Parameter(self.wy, requires_grad=False)

    def forward(self, x):
        s = list(x.size())
        scoremap = self.beta(x)
        scoremap = scoremap.view([s[0], s[1], s[2] * s[3]])
        scoremap = F.softmax(scoremap, dim=2)
        scoremap = scoremap.view([s[0], s[1], s[2], s[3]])
        scoremap_x = scoremap.mul(self.wx)
        scoremap_x = scoremap_x.view([s[0], s[1], s[2] * s[3]])
        soft_argmax_x = torch.sum(scoremap_x, dim=2)
        scoremap_y = scoremap.mul(self.wy)
        scoremap_y = scoremap_y.view([s[0], s[1], s[2] * s[3]])
        soft_argmax_y = torch.sum(scoremap_y, dim=2)
        keypoint_uv = torch.stack([soft_argmax_x, soft_argmax_y], dim=2)
        return keypoint_uv, scoremap


class OfficialResNetUnet_RGB(nn.Module):
    def __init__(self, backbone, joint_num, pretrain=True, deconv_dim=128):
        super(OfficialResNetUnet_RGB, self).__init__()
        self.joint_num = joint_num
        self.feature_dim = [self.joint_num * 3, self.joint_num]
        layers_num = int(backbone.split('-')[-1])
        block, layers = resnet[layers_num]
        self.backbone = ResNet(block, layers)
        self.skip_layer4 = Residual(256*block.expansion, 256)
        self.up4 = nn.Sequential(Residual(512*block.expansion, 512),
                                 nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer4 = Residual((512+256), 256)

        self.skip_layer3 = Residual(128*block.expansion, 128)
        self.up3 = nn.Sequential(Residual(256, 256),
                                  nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer3 = Residual((256+128), 128)

        self.skip_layer2 = Residual(64*block.expansion, 64)
        self.up2 = nn.Sequential(Residual(128, 128),
                                  nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer2 = Residual((128+64), deconv_dim)

        self.finals = nn.ModuleList()
        self.heatmap_size=32
        self.soft_heatmap = SoftHeatmap(self.heatmap_size, joint_num)
        self.pool = nn.AvgPool2d(2, stride=2)
        out_dim_list = [joint_num*2,joint_num, joint_num, joint_num]#offsetsy,hxy,hz,wm
        for out_dim in out_dim_list:
            self.finals.append(nn.Conv2d(in_channels=deconv_dim, out_channels=out_dim, kernel_size=1, stride=1))

        self.init_weights()
        if pretrain:
            if layers_num == 18:
                print('load weight from resnet-18')
                pretrain_weight = torchvision.models.resnet18(pretrained=True)
            elif layers_num == 50:
                print('load weight from resnet-50')
                pretrain_weight = torchvision.models.resnet50(pretrained=True)
            elif layers_num == 101:
                print('load weight from resnet-101')
                pretrain_weight = torchvision.models.resnet101(pretrained=True)
            self.backbone.load_state_dict(pretrain_weight.state_dict(), strict=False)
        #self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)

        for m in self.finals.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, img):
        device = img.device
        c0, c1, c2, c3, c4 = self.backbone(img)

        c4_up = self.up4(c4)
        c3_skip = self.skip_layer4(c3)
        c3_fusion = self.fusion_layer4(torch.cat((c4_up, c3_skip), dim=1))

        c3_up = self.up3(c3_fusion)
        c2_skip = self.skip_layer3(c2)
        c2_fusion = self.fusion_layer3(torch.cat((c3_up, c2_skip), dim=1))

        c2_up = self.up2(c2_fusion)
        c1_skip = self.skip_layer2(c1)
        img_feature = self.fusion_layer2(torch.cat((c2_up, c1_skip), dim=1)) #B*128*64*64

        #img_feature = self.pool(img_feature)

        offsetxy = self.finals[0](img_feature)
        heatmap_xy =self.finals[1](img_feature)
        heatmap_z = self.finals[2](img_feature)
        weight_map = self.finals[3](img_feature)

        coord_xy, softheatmap_xy = self.soft_heatmap(heatmap_xy)
        depth_maps = softheatmap_xy * heatmap_z
        coord_z = torch.sum(depth_maps.view(-1, self.joint_num, depth_maps.shape[2] * depth_maps.shape[3]), dim=2, keepdim=True)
        # B*(J*3)*64*64
        img_result = torch.cat((offsetxy,heatmap_xy,weight_map),1)
        #print(img_result.shape)
        return img_result, img_feature,coord_z

class OfficialResNetUnet_depth2latent(nn.Module):
    def __init__(self, backbone, joint_num, pretrain=True, deconv_dim=128):
        super(OfficialResNetUnet_depth2latent, self).__init__()
        self.joint_num = joint_num
        self.feature_dim = [self.joint_num * 3, self.joint_num]
        layers_num = int(backbone.split('-')[-1])
        block, layers = resnet[layers_num]
        self.backbone_rgb = ResNet(block, layers)
        self.backbone_depth = ResNet(block, layers)
        self.skip_layer4 = Residual(256*block.expansion, 256)
        self.up4 = nn.Sequential(Residual(512*block.expansion, 512),
                                 nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer4 = Residual((512+256), 256)

        self.skip_layer3 = Residual(128*block.expansion, 128)
        self.up3 = nn.Sequential(Residual(256, 256),
                                  nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer3 = Residual((256+128), 128)

        self.skip_layer2 = Residual(64*block.expansion, 64)
        self.up2 = nn.Sequential(Residual(128, 128),
                                  nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer2 = Residual((128+64), deconv_dim)

        self.skip_layer4_d = Residual(256 * block.expansion, 256)
        self.up4_d = nn.Sequential(Residual(512 * block.expansion, 512),
                                 nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer4_d  = Residual((512 + 256), 256)

        self.skip_layer3_d  = Residual(128 * block.expansion, 128)
        self.up3_d  = nn.Sequential(Residual(256, 256),
                                 nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer3_d  = Residual((256 + 128), 128)

        self.skip_layer2_d  = Residual(64 * block.expansion, 64)
        self.up2_d  = nn.Sequential(Residual(128, 128),
                                 nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer2_d  = Residual((128 + 64), deconv_dim)

        self.finals = nn.ModuleList()
        self.heatmap_size=32
        self.soft_heatmap = SoftHeatmap(self.heatmap_size, joint_num)
        self.pool = nn.AvgPool2d(2, stride=2)
        out_dim_list = [joint_num*2,joint_num, joint_num, joint_num]#offsetsy,hxy,hz,wm
        for out_dim in out_dim_list:
            self.finals.append(nn.Conv2d(in_channels=deconv_dim, out_channels=out_dim, kernel_size=1, stride=1))

        self.init_weights()
        if pretrain:
            if layers_num == 18:
                print('load weight from resnet-18')
                pretrain_weight = torchvision.models.resnet18(pretrained=True)
            elif layers_num == 50:
                print('load weight from resnet-50')
                pretrain_weight = torchvision.models.resnet50(pretrained=True)
            elif layers_num == 101:
                print('load weight from resnet-101')
                pretrain_weight = torchvision.models.resnet101(pretrained=True)
            self.backbone_rgb.load_state_dict(pretrain_weight.state_dict(), strict=False)
            self.backbone_depth.load_state_dict(pretrain_weight.state_dict(), strict=False)
        self.backbone_depth.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)

        for m in self.finals.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, rgb,depth):
        device = rgb.device
        c0, c1, c2, c3, c4 = self.backbone_rgb(rgb)
        c4_up = self.up4(c4)
        c3_skip = self.skip_layer4(c3)
        c3_fusion = self.fusion_layer4(torch.cat((c4_up, c3_skip), dim=1))

        c3_up = self.up3(c3_fusion)
        c2_skip = self.skip_layer3(c2)
        c2_fusion = self.fusion_layer3(torch.cat((c3_up, c2_skip), dim=1))

        c2_up = self.up2(c2_fusion)
        c1_skip = self.skip_layer2(c1)
        img_feature = self.fusion_layer2(torch.cat((c2_up, c1_skip), dim=1))

        c0_d, c1_d, c2_d, c3_d, c4_d = self.backbone_depth(depth)
        c4_up_d = self.up4_d(c4_d)
        c3_skip_d = self.skip_layer4_d(c3_d)
        c3_fusion_d = self.fusion_layer4_d(torch.cat((c4_up_d, c3_skip_d), dim=1))

        c3_up_d = self.up3_d(c3_fusion_d)
        c2_skip_d = self.skip_layer3_d(c2_d)
        c2_fusion_d = self.fusion_layer3_d(torch.cat((c3_up_d, c2_skip_d), dim=1))

        c2_up_d = self.up2_d(c2_fusion_d)
        c1_skip_d = self.skip_layer2_d(c1_d)
        img_feature_d = self.fusion_layer2_d(torch.cat((c2_up_d, c1_skip_d), dim=1))

        offsetxy = self.finals[0](img_feature)
        heatmap_xy =self.finals[1](img_feature)
        heatmap_z = self.finals[2](img_feature_d)
        weight_map = self.finals[3](img_feature)

        coord_xy, softheatmap_xy = self.soft_heatmap(heatmap_xy)
        depth_maps = softheatmap_xy * heatmap_z
        coord_z = torch.sum(depth_maps.view(-1, self.joint_num, depth_maps.shape[2] * depth_maps.shape[3]), dim=2, keepdim=True)
        # B*(J*3)*64*64
        img_result = torch.cat((offsetxy,heatmap_xy,weight_map),1)
        #print(img_result.shape)
        return img_result, img_feature,coord_z


class OfficialResNetUnet_RGBD(nn.Module):
    def __init__(self, backbone, joint_num, pretrain=True, deconv_dim=128, out_dim_list=[3*21, 21, 21]):
        super(OfficialResNetUnet_RGBD, self).__init__()
        self.joint_num = joint_num
        self.feature_dim = [self.joint_num * 3, self.joint_num]
        layers_num = int(backbone.split('-')[-1])
        block, layers = resnet_rgbd[layers_num]
        backbone = RGBD_ResNet(block, layers)

        self.skip_layer4 = Residual(256*block.expansion, 256)
        self.up4 = nn.Sequential(Residual(512*block.expansion, 512),
                                 nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer4 = Residual((512+256), 256)

        self.skip_layer3 = Residual(128*block.expansion, 128)
        self.up3 = nn.Sequential(Residual(256, 256),
                                  nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer3 = Residual((256+128), 128)

        self.skip_layer2 = Residual(64*block.expansion, 64)
        self.up2 = nn.Sequential(Residual(128, 128),
                                  nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer2 = Residual((128+64), deconv_dim)

        self.finals = nn.ModuleList()
        for out_dim in out_dim_list:
            self.finals.append(nn.Conv2d(in_channels=deconv_dim, out_channels=out_dim, kernel_size=1, stride=1))

        self.init_weights()

        if pretrain:
            if layers_num == 18:
                print('load Pretrained_weight from resnet-18')
                pretrain_weight = torchvision.models.resnet18(pretrained=True)
            elif layers_num == 34:
                print('load Pretrained_weight from resnet-34')
                pretrain_weight = torchvision.models.resnet34(pretrained=True)
            elif layers_num == 50:
                print('load Pretrained_weight from resnet-50')
                pretrain_weight = torchvision.models.resnet50(pretrained=True)
            elif layers_num == 101:
                print('load Pretrained_weight from resnet-101')
                pretrain_weight = torchvision.models.resnet101(pretrained=True)
            # self.backbone.load_state_dict(pretrain_weight.state_dict(), strict=False)
            self.backbone = load_dualpath_model(backbone,pretrain_weight.state_dict())
        else:
            self.backbone = backbone

        #self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.depth_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)

        for m in self.finals.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, img_rgb,img_depth,writer=None,ii=0):
        device = img_depth.device
        blocks,merges = self.backbone(img_rgb, img_depth,writer,ii)
        #c1-4,m1-4
        #c0, c1, c2, c3, c4 = self.backbone(img_rgb,img_depth)

        c4_up = self.up4(merges[3])
        c3_skip = self.skip_layer4(merges[2])
        c3_fusion = self.fusion_layer4(torch.cat((c4_up, c3_skip), dim=1))

        c3_up = self.up3(c3_fusion)
        c2_skip = self.skip_layer3(merges[1])
        c2_fusion = self.fusion_layer3(torch.cat((c3_up, c2_skip), dim=1))

        c2_up = self.up2(c2_fusion)
        c1_skip = self.skip_layer2(merges[0])
        img_feature = self.fusion_layer2(torch.cat((c2_up, c1_skip), dim=1))#B*128*32*32

        img_result = torch.Tensor().to(device)
        for layer in self.finals:
            temp = layer(img_feature)
            img_result = torch.cat((img_result, temp), dim=1)

        return img_result, img_feature

class OfficialResNetUnet_Inject(nn.Module):
    def __init__(self, backbone, joint_num, pretrain=True, deconv_dim=128, out_dim_list=[3*21, 21, 21]):
        super(OfficialResNetUnet_Inject, self).__init__()
        self.joint_num = joint_num
        self.feature_dim = [self.joint_num * 3, self.joint_num]
        layers_num = int(backbone.split('-')[-1])
        block, layers = resnet_rgbd[layers_num]
        backbone = RGBD_ResNet_Inject(block, layers)

        self.skip_layer4 = Residual(256*block.expansion, 256)
        self.up4 = nn.Sequential(Residual(512*block.expansion, 512),
                                 nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer4 = Residual((512+256), 256)

        self.skip_layer3 = Residual(128*block.expansion, 128)
        self.up3 = nn.Sequential(Residual(256, 256),
                                  nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer3 = Residual((256+128), 128)

        self.skip_layer2 = Residual(64*block.expansion, 64)
        self.up2 = nn.Sequential(Residual(128, 128),
                                  nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer2 = Residual((128+64), deconv_dim)

        self.finals = nn.ModuleList()
        for out_dim in out_dim_list:
            self.finals.append(nn.Conv2d(in_channels=deconv_dim, out_channels=out_dim, kernel_size=1, stride=1))

        self.init_weights()

        if pretrain:
            if layers_num == 18:
                print('load Pretrained_weight from resnet-18')
                pretrain_weight = torchvision.models.resnet18(pretrained=True)
            elif layers_num == 34:
                print('load Pretrained_weight from resnet-34')
                pretrain_weight = torchvision.models.resnet34(pretrained=True)
            elif layers_num == 50:
                print('load Pretrained_weight from resnet-50')
                pretrain_weight = torchvision.models.resnet50(pretrained=True)
            elif layers_num == 101:
                print('load Pretrained_weight from resnet-101')
                pretrain_weight = torchvision.models.resnet101(pretrained=True)
            # self.backbone.load_state_dict(pretrain_weight.state_dict(), strict=False)
            self.backbone = load_dualpath_model(backbone,pretrain_weight.state_dict())
        else:
            self.backbone = backbone

        #self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.depth_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)

        for m in self.finals.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, img_rgb,img_depth,writer=None,ii=0):
        device = img_depth.device
        blocks, merges = self.backbone(img_rgb, img_depth, writer, ii)
        # c1-4,m1-4
        # c0, c1, c2, c3, c4 = self.backbone(img_rgb,img_depth)

        c4_up = self.up4(merges[3])
        c3_skip = self.skip_layer4(merges[2])
        c3_fusion = self.fusion_layer4(torch.cat((c4_up, c3_skip), dim=1))

        c3_up = self.up3(c3_fusion)
        c2_skip = self.skip_layer3(merges[1])
        c2_fusion = self.fusion_layer3(torch.cat((c3_up, c2_skip), dim=1))

        c2_up = self.up2(c2_fusion)
        c1_skip = self.skip_layer2(merges[0])
        img_feature = self.fusion_layer2(torch.cat((c2_up, c1_skip), dim=1))  # B*128*32*32

        img_result = torch.Tensor().to(device)
        for layer in self.finals:
            temp = layer(img_feature)
            img_result = torch.cat((img_result, temp), dim=1)

        return img_result, img_feature


class OfficialResNetUnet_RGBD2latent(nn.Module):
    def __init__(self, backbone, joint_num, pretrain=True, deconv_dim=128):
        super(OfficialResNetUnet_RGBD2latent, self).__init__()
        self.joint_num = joint_num
        self.feature_dim = [self.joint_num * 3, self.joint_num]
        layers_num = int(backbone.split('-')[-1])
        block, layers = resnet_rgbd[layers_num]
        backbone = RGBD_ResNet(block, layers)

        self.skip_layer4 = Residual(256*block.expansion, 256)
        self.up4 = nn.Sequential(Residual(512*block.expansion, 512),
                                 nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer4 = Residual((512+256), 256)

        self.skip_layer3 = Residual(128*block.expansion, 128)
        self.up3 = nn.Sequential(Residual(256, 256),
                                  nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer3 = Residual((256+128), 128)

        self.skip_layer2 = Residual(64*block.expansion, 64)
        self.up2 = nn.Sequential(Residual(128, 128),
                                  nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer2 = Residual((128+64), deconv_dim)

        self.finals = nn.ModuleList()
        self.heatmap_size = 32
        self.soft_heatmap = SoftHeatmap(self.heatmap_size, joint_num)
        out_dim_list = [joint_num * 2, joint_num, joint_num, joint_num]  # offsetsy,hxy,hz,wm
        for out_dim in out_dim_list:
            self.finals.append(nn.Conv2d(in_channels=deconv_dim, out_channels=out_dim, kernel_size=1, stride=1))

        self.init_weights()

        if pretrain:
            if layers_num == 18:
                print('load Pretrained_weight from resnet-18')
                pretrain_weight = torchvision.models.resnet18(pretrained=True)
            elif layers_num == 34:
                print('load Pretrained_weight from resnet-34')
                pretrain_weight = torchvision.models.resnet34(pretrained=True)
            elif layers_num == 50:
                print('load Pretrained_weight from resnet-50')
                pretrain_weight = torchvision.models.resnet50(pretrained=True)
            elif layers_num == 101:
                print('load Pretrained_weight from resnet-101')
                pretrain_weight = torchvision.models.resnet101(pretrained=True)
            # self.backbone.load_state_dict(pretrain_weight.state_dict(), strict=False)
            self.backbone = load_dualpath_model(backbone,pretrain_weight.state_dict())
        else:
            self.backbone = backbone

        #self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.depth_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)

        for m in self.finals.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, img_rgb,img_depth,writer=None,ii=0):
        device = img_depth.device
        blocks,merges = self.backbone(img_rgb, img_depth,writer,ii)
        #c1-4,m1-4
        #c0, c1, c2, c3, c4 = self.backbone(img_rgb,img_depth)

        c4_up = self.up4(merges[3])
        c3_skip = self.skip_layer4(merges[2])
        c3_fusion = self.fusion_layer4(torch.cat((c4_up, c3_skip), dim=1))

        c3_up = self.up3(c3_fusion)
        c2_skip = self.skip_layer3(merges[1])
        c2_fusion = self.fusion_layer3(torch.cat((c3_up, c2_skip), dim=1))

        c2_up = self.up2(c2_fusion)
        c1_skip = self.skip_layer2(merges[0])
        img_feature = self.fusion_layer2(torch.cat((c2_up, c1_skip), dim=1))#B*128*32*32

        offsetxy = self.finals[0](img_feature)
        heatmap_xy = self.finals[1](img_feature)
        heatmap_z = self.finals[2](img_feature)
        weight_map = self.finals[3](img_feature)

        coord_xy, softheatmap_xy = self.soft_heatmap(heatmap_xy)
        depth_maps = softheatmap_xy * heatmap_z
        coord_z = torch.sum(depth_maps.view(-1, self.joint_num, depth_maps.shape[2] * depth_maps.shape[3]), dim=2,
                            keepdim=True)
        # B*(J*3)*64*64
        img_result = torch.cat((offsetxy, heatmap_xy, weight_map), 1)
        # print(img_result.shape)
        return img_result, img_feature, coord_z


class OfficialResNetUnet_Supervision_Fusion(nn.Module):
    def __init__(self, backbone, joint_num, pretrain=True, deconv_dim=128, out_dim_list=[3*21, 21, 21]):
        super(OfficialResNetUnet_Supervision_Fusion, self).__init__()
        self.joint_num = joint_num
        self.feature_dim = [self.joint_num * 3, self.joint_num]
        layers_num = int(backbone.split('-')[-1])
        block, layers = resnet_rgbd[layers_num]
        backbone_rgb = ResNet(block, layers)
        backbone_d = ResNet(block, layers)

        self.skip_layer4 = Residual(256*block.expansion, 256)
        self.up4 = nn.Sequential(Residual(512*block.expansion, 512),
                                 nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer4 = Residual((512+256), 256)

        self.skip_layer3 = Residual(128*block.expansion, 128)
        self.up3 = nn.Sequential(Residual(256, 256),
                                  nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer3 = Residual((256+128), 128)

        self.skip_layer2 = Residual(64*block.expansion, 64)
        self.up2 = nn.Sequential(Residual(128, 128),
                                  nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer2 = Residual((128+64), deconv_dim)

        self.skip_layer4_d = Residual(256 * block.expansion, 256)
        self.up4_d = nn.Sequential(Residual(512 * block.expansion, 512),
                                 nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer4_d = Residual((512 + 256), 256)

        self.skip_layer3_d = Residual(128 * block.expansion, 128)
        self.up3_d = nn.Sequential(Residual(256, 256),
                                 nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer3_d = Residual((256 + 128), 128)

        self.skip_layer2_d = Residual(64 * block.expansion, 64)
        self.up2_d = nn.Sequential(Residual(128, 128),
                                 nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer2_d = Residual((128 + 64), deconv_dim)

        self.finals = nn.ModuleList()
        for out_dim in out_dim_list:
            self.finals.append(nn.Conv2d(in_channels=deconv_dim, out_channels=out_dim, kernel_size=1, stride=1))

        self.init_weights()

        if pretrain:
            if layers_num == 18:
                print('load Pretrained_weight from resnet-18')
                pretrain_weight = torchvision.models.resnet18(pretrained=True)
            elif layers_num == 34:
                print('load Pretrained_weight from resnet-34')
                pretrain_weight = torchvision.models.resnet34(pretrained=True)
            elif layers_num == 50:
                print('load Pretrained_weight from resnet-50')
                pretrain_weight = torchvision.models.resnet50(pretrained=True)
            elif layers_num == 101:
                print('load Pretrained_weight from resnet-101')
                pretrain_weight = torchvision.models.resnet101(pretrained=True)
            # self.backbone.load_state_dict(pretrain_weight.state_dict(), strict=False)
            self.backbone_rgb = load_dualpath_model(backbone_rgb,pretrain_weight.state_dict())
            self.backbone_d = load_dualpath_model(backbone_d, pretrain_weight.state_dict())
        else:
            self.backbone_rgb = backbone_rgb
            self.backbone_d = backbone_d

        #self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone_d.depth_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)

        for m in self.finals.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self,img_depth,img_rgb,writer=None,ii=0):
        device = img_depth.device
        c0, c1, c2, c3, c4 = self.backbone_rgb(img_rgb)
        c4_up = self.up4(c4)
        c3_skip = self.skip_layer4(c3)
        c3_fusion = self.fusion_layer4(torch.cat((c4_up, c3_skip), dim=1))

        c3_up = self.up3(c3_fusion)
        c2_skip = self.skip_layer3(c2)
        c2_fusion = self.fusion_layer3(torch.cat((c3_up, c2_skip), dim=1))

        c2_up = self.up2(c2_fusion)
        c1_skip = self.skip_layer2(c1)
        img_feature = self.fusion_layer2(torch.cat((c2_up, c1_skip), dim=1))

        c0_d, c1_d, c2_d, c3_d, c4_d = self.backbone_d(img_depth)
        c4_up_d = self.up4_d(c4_d)
        c3_skip_d = self.skip_layer4_d(c3_d)
        c3_fusion_d = self.fusion_layer4_d(torch.cat((c4_up_d, c3_skip_d), dim=1))

        c3_up_d = self.up3_d(c3_fusion_d)
        c2_skip_d = self.skip_layer3_d(c2_d)
        c2_fusion_d = self.fusion_layer3_d(torch.cat((c3_up_d, c2_skip_d), dim=1))

        c2_up_d = self.up2_d(c2_fusion_d)
        c1_skip_d = self.skip_layer2_d(c1_d)
        img_feature_d = self.fusion_layer2_d(torch.cat((c2_up_d, c1_skip_d), dim=1))

        #fusion


        img_result = torch.Tensor().to(device)
        for layer in self.finals:
            temp = layer(img_feature)
            img_result = torch.cat((img_result, temp), dim=1)

        return img_result, img_feature

if __name__ == '__main__':
    input_img = torch.rand([2, 1, 128, 128])
    model = OfficialResNetUnet('resnet-18', 21, deconv_dim=128)
    re,fs = model(input_img)
    print(re.shape)
    print(fs.shape)